"""Training mechanics for one layer's routing predictor (train-all variant).

Each function owns one step of a single layer's run, so the remote trainer's
per-layer loop reads as a short sequence of calls. torch is imported inside the
functions (never at module scope), so the module can be imported by the local
entrypoint without pulling the GPU stack.
"""

from __future__ import annotations

import math
import time
from typing import Any, NamedTuple

from helper import (
    kl_divergence,
    load_cache_frames,
    load_cached_expert_counts,
    resident_sets,
    score_metrics,
)


class Frame(NamedTuple):
    """One split's loaded rows: ``x`` inputs, ``y`` scores, ``ids`` true top-k."""

    x: Any
    y: Any
    ids: Any


def resident_report(
    cache_dir: str,
    manifest: dict[str, Any],
    layer: int,
    ratios: tuple[float, ...],
    n_experts: int,
) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    """Hottest-k resident sets for ``layer`` plus the hot-experts artifact."""
    layer_counts = load_cached_expert_counts(cache_dir, manifest, layer)
    resident = resident_sets(layer_counts, ratios)
    counts_entry = manifest["entries"][f"expert_counts:{layer}"]
    print(
        f"ready sets: {counts_entry['source']} layer {layer} "
        + ", ".join(f"{int(r * 100)}%={resident[r]['k']}" for r in ratios),
        flush=True,
    )
    hot_experts = {
        "layer": layer,
        "source": counts_entry["source"],
        "n_experts": n_experts,
        "ratios": {
            str(r): {"k": resident[r]["k"], "experts": resident[r]["ids"]}
            for r in ratios
        },
    }
    return resident, hot_experts


def load_frames(
    cfg: dict[str, Any], manifest: dict[str, Any], task: dict[str, Any], seed: int
) -> tuple[Frame, Frame]:
    """Load the train and eval frames from the contiguous cache."""
    cache_dir = cfg["cache"]["dir"]
    train_cfg = cfg["data"]["train"]
    eval_cfg = cfg["data"]["eval"]
    started = time.monotonic()
    train = Frame(
        *load_cache_frames(
            cache_dir,
            manifest,
            "train",
            task["source_layer"],
            task["layer"],
            task["input_kind"],
            train_cfg["datasets"],
            train_cfg["max_slices"],
            train_cfg["max_tokens"],
            seed,
        )
    )
    ev = Frame(
        *load_cache_frames(
            cache_dir,
            manifest,
            eval_cfg["split"],
            task["source_layer"],
            task["layer"],
            task["input_kind"],
            eval_cfg["datasets"],
            None,
            eval_cfg["max_tokens"],
            seed + 1,
        )
    )
    print(f"[data] loaded cache in {time.monotonic() - started:.1f}s", flush=True)
    return train, ev


def build_training(
    cfg: dict[str, Any], dims: dict[str, int], d_in: int, n_train: int
) -> tuple[Any, Any, Any]:
    """Build the predictor, its optimizer, and the optional LR scheduler."""
    import torch
    from models import build_model

    optim_cfg = cfg["optim"]
    model = build_model(
        cfg["model"]["arch"],
        d_in,
        dims["n_experts"],
        rank=cfg["model"]["rank"],
        hidden=cfg["model"]["hidden"],
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg["weight_decay"]),
    )
    steps_per_epoch = math.ceil(n_train / int(optim_cfg["batch_size"]))
    scheduler = None
    if optim_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, int(optim_cfg["epochs"]) * steps_per_epoch)
        )
    return model, optimizer, scheduler


def evaluate(
    model: Any,
    frame: Frame,
    ks: list[int],
    bias: Any,
    resident: dict[float, dict[str, Any]],
    chunk: int,
) -> dict[str, float]:
    """Forward the eval frame in chunks and score it against the reference."""
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, int(frame.x.shape[0]), chunk):
            xb = frame.x[start : start + chunk].cuda().float()
            preds.append(model(xb).float().cpu())
    return score_metrics(
        torch.cat(preds), frame.y, frame.ids, ks, bias=bias, resident=resident
    )


def run_epochs(
    cfg: dict[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any,
    train: Frame,
    ks: list[int],
    ratios: tuple[float, ...],
    evaluate: Any,
) -> tuple[list[dict], Any, int, float]:
    """Train for ``optim.epochs``, evaluating every ``eval.every`` epochs.

    Returns ``(history, best_state, best_epoch, best_eval_kl)``; ``best_state``
    is ``None`` when no eval ran or ``output.save_best`` is false.
    """
    import torch

    optim_cfg = cfg["optim"]
    epochs = int(optim_cfg["epochs"])
    batch_size = int(optim_cfg["batch_size"])
    every = max(1, int(cfg["eval"]["every"]))
    save_best = bool(cfg["output"]["save_best"])
    n_train = int(train.x.shape[0])

    history: list[dict] = []
    best_eval_kl = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        perm = torch.randperm(n_train)
        running = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = train.x[idx].cuda().float()
            yb = train.y[idx].cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = kl_divergence(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optim_cfg["grad_clip"])
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]

        train_kl = running / max(seen, 1)
        entry: dict = {
            "epoch": epoch + 1,
            "train_kl": round(train_kl, 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        do_eval = ((epoch + 1) % every == 0) or (epoch + 1 == epochs)
        parts = [
            f"epoch {epoch + 1}/{epochs}",
            f"train_kl={train_kl:.5f}",
            f"lr={entry['lr']:.2e}",
        ]
        if do_eval:
            metrics = evaluate()
            entry["eval_kl"] = round(metrics["kl"], 6)
            for k in ks:
                entry[f"recall@{k}"] = round(metrics[f"recall@{k}"], 6)
            for r in ratios:
                tag = f"r{int(round(r * 100))}"
                entry[f"resident_recall_{tag}"] = round(
                    metrics[f"resident_recall_{tag}"], 6
                )
                for k in ks:
                    entry[f"ready_recall_{tag}@{k}"] = round(
                        metrics[f"ready_recall_{tag}@{k}"], 6
                    )
            parts.append(f"eval_kl={metrics['kl']:.5f}")
            for k in ks:
                parts.append(f"recall@{k}={metrics[f'recall@{k}']:.4f}")
            first_k = ks[0]
            ready = " ".join(
                f"{int(r * 100)}%="
                f"{metrics[f'ready_recall_r{int(round(r * 100))}@{first_k}']:.3f}"
                for r in ratios
            )
            parts.append(f"ready@{first_k}[{ready}]")
            if metrics["kl"] < best_eval_kl:
                best_eval_kl = metrics["kl"]
                best_epoch = epoch + 1
                if save_best:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
        entry["seconds"] = round(time.monotonic() - started, 2)
        parts.append(f"{entry['seconds']:.1f}s")
        history.append(entry)
        print(" ".join(parts), flush=True)

    return history, best_state, best_epoch, best_eval_kl


def save_checkpoint(model: Any, path: str) -> None:
    """Write the model state to a safetensors checkpoint."""
    from safetensors.torch import save_file

    save_file(
        {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        },
        path,
    )


def build_summary(
    run_id: str,
    row_id: str,
    cfg: dict[str, Any],
    dims: dict[str, int],
    task: dict[str, Any],
    d_in: int,
    model: Any,
    n_train: int,
    n_eval: int,
    best_epoch: int,
    best_eval_kl: float,
    final: dict[str, float],
    history: list[dict],
) -> dict[str, Any]:
    """Assemble one layer's summary written to its ``metrics.json``."""
    return {
        "run_id": run_id,
        "row_id": row_id,
        "config": cfg,
        "n_layers": dims["n_layers"],
        "n_hash_layers": dims["n_hash_layers"],
        "n_experts": dims["n_experts"],
        "layer": task["layer"],
        "source_layer": task["source_layer"],
        "distance": task["distance"],
        "d_in": d_in,
        "params": sum(p.numel() for p in model.parameters()),
        "n_train_tokens": n_train,
        "n_eval_tokens": n_eval,
        "best_epoch": best_epoch,
        "best_eval_kl": None if best_epoch < 0 else round(best_eval_kl, 6),
        "final_metrics": final,
        "history": history,
    }
