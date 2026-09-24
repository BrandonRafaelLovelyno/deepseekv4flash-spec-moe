"""Single-variant routing-predictor prototype with per-epoch evaluation.

Trains **one** predictor: for a target layer ``L`` it reads the activation of
layer ``L - distance`` and predicts layer ``L``'s 256-way score vector. Every
``eval.every`` epochs it evaluates on the held-out split, reporting both the KL
loss (train vs eval, for hyper-parameter tuning) and the practical expert-selection
recall, so overfitting is visible as it starts.

All knobs live in ``5_train_one/config.yaml`` (copy ``config.example.yaml``).
The local entrypoint reads it as text and hands it to the remote job; the remote
parses it with PyYAML.

Artifacts land under ``<output.volume_dir>/<run_id>/`` on the
``deepseek-v4-flash-training`` volume and are mirrored to
``5_train_one/output/<run_id>/``.

Usage:
    modal run 5_train_one/main.py
    modal run 5_train_one/main.py --config 5_train_one/other.yaml
"""

import json
import math
import os
import sys
import time

import modal

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from helper import (
    HARVEST_DIR,
    TRAINING_DIR,
    _dims,
    _manifest,
    _select_records,
    kl_divergence,
    load_config,
    load_frames,
    load_router_bias,
    score_metrics,
)

APP_NAME = "deepseek-v4-flash-train-one"

OUT_DIR = os.path.join(THIS_DIR, "output")

harvest_vol = modal.Volume.from_name(
    "deepseek-v4-flash-harvest", create_if_missing=True
)
training_vol = modal.Volume.from_name(
    "deepseek-v4-flash-training", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "safetensors", "numpy", "matplotlib", "pyyaml")
    .env({"MPLBACKEND": "Agg"})
    .add_local_python_source("helper", "models")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu="L4",
    volumes={HARVEST_DIR: harvest_vol, TRAINING_DIR: training_vol},
    cpu=8,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
)
def train_variant(config_text: str, run_id: str) -> dict:
    """Train one predictor and evaluate it at the end of every epoch."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from models import build_model
    from safetensors.torch import save_file

    cfg = load_config(config_text)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.cuda.set_device(0)

    task = cfg["task"]
    layer = int(task["layer"])
    distance = int(task["distance"])
    input_kind = task["input"]
    arch = cfg["model"]["arch"]

    harvest_dir = cfg["data"]["harvest_dir"]
    manifest = _manifest(harvest_dir)
    dims = _dims(manifest)
    if layer >= dims["n_layers"]:
        raise ValueError(f"layer {layer} >= n_layers {dims['n_layers']}")
    if layer < dims["n_hash_layers"]:
        raise ValueError(f"layer {layer} is a hash layer (token-id routing)")
    source_layer = layer - distance
    if source_layer < 0:
        raise ValueError(f"layer {layer} has no source at distance {distance}")

    # The reference selects experts by top-k(score + bias); the bias is applied
    # only at selection, never to the pre-bias score the model is trained on.
    layer_bias = load_router_bias(manifest, harvest_dir)[layer]

    train_cfg = cfg["data"]["train"]
    eval_cfg = cfg["data"]["eval"]
    train_records = _select_records(
        manifest, "train", train_cfg["datasets"], train_cfg["max_slices"], seed
    )
    eval_records = _select_records(
        manifest, eval_cfg["split"], eval_cfg["datasets"], None, seed
    )
    if not train_records:
        raise ValueError("no training records selected")
    if not eval_records:
        raise ValueError("no evaluation records selected")

    x, y, _ = load_frames(
        train_records, source_layer, layer, input_kind, train_cfg["max_tokens"], seed
    )
    eval_x, eval_y, eval_ids = load_frames(
        eval_records, source_layer, layer, input_kind, eval_cfg["max_tokens"], seed + 1
    )
    n_train = int(x.shape[0])
    n_eval = int(eval_x.shape[0])
    d_in = int(x.shape[1])

    optim_cfg = cfg["optim"]
    epochs = int(optim_cfg["epochs"])
    batch_size = int(optim_cfg["batch_size"])
    ks = [int(k) for k in cfg["eval"]["ks"]]
    every = max(1, int(cfg["eval"]["every"]))
    chunk = int(cfg["eval"]["chunk_tokens"])

    model = build_model(
        arch,
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
    steps_per_epoch = math.ceil(n_train / batch_size)
    scheduler = None
    if optim_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(1, epochs * steps_per_epoch)
        )

    def evaluate() -> dict[str, float]:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, n_eval, chunk):
                xb = eval_x[start : start + chunk].cuda().float()
                preds.append(model(xb).float().cpu())
        return score_metrics(torch.cat(preds), eval_y, eval_ids, ks, bias=layer_bias)

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
            xb = x[idx].cuda().float()
            yb = y[idx].cuda()
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
            parts.append(f"eval_kl={metrics['kl']:.5f}")
            for k in ks:
                parts.append(f"recall@{k}={metrics[f'recall@{k}']:.4f}")
            if metrics["kl"] < best_eval_kl:
                best_eval_kl = metrics["kl"]
                best_epoch = epoch + 1
                if cfg["output"]["save_best"]:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
        entry["seconds"] = round(time.monotonic() - started, 2)
        parts.append(f"{entry['seconds']:.1f}s")
        history.append(entry)
        print(" ".join(parts), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate()

    row_id = f"L{layer:02d}_{arch}_d{distance}_{input_kind}"
    out_dir = os.path.join(cfg["output"]["volume_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)
    save_file(
        {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        },
        os.path.join(out_dir, "checkpoint.safetensors"),
    )
    summary = {
        "run_id": run_id,
        "row_id": row_id,
        "config": cfg,
        "n_layers": dims["n_layers"],
        "n_hash_layers": dims["n_hash_layers"],
        "n_experts": dims["n_experts"],
        "source_layer": source_layer,
        "d_in": d_in,
        "params": sum(p.numel() for p in model.parameters()),
        "n_train_tokens": n_train,
        "n_eval_tokens": n_eval,
        "best_epoch": best_epoch,
        "best_eval_kl": None if best_epoch < 0 else round(best_eval_kl, 6),
        "final_metrics": final,
        "history": history,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    training_vol.commit()

    fig, (ax_loss, ax_recall) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax_loss.plot(
        [h["epoch"] for h in history], [h["train_kl"] for h in history], label="train"
    )
    eval_epochs = [h["epoch"] for h in history if "eval_kl" in h]
    ax_loss.plot(
        eval_epochs,
        [h["eval_kl"] for h in history if "eval_kl" in h],
        "-o",
        markersize=3,
        label="eval",
    )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("KL divergence")
    ax_loss.set_title(row_id)
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    for k in ks:
        ax_recall.plot(
            eval_epochs,
            [h[f"recall@{k}"] for h in history if f"recall@{k}" in h],
            "-o",
            markersize=3,
            label=f"recall@{k}",
        )
    ax_recall.set_xlabel("epoch")
    ax_recall.set_ylabel("recall")
    ax_recall.set_ylim(0.0, 1.0)
    ax_recall.legend()
    ax_recall.grid(alpha=0.3)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "run_id": run_id,
        "row_id": row_id,
        "summary": summary,
        "png": buffer.getvalue(),
    }


def _write_history_csv(path: str, history: list[dict], ks: list[int]) -> None:
    """Write the per-epoch history to CSV, leaving skipped eval cells empty."""
    columns = (
        ["epoch", "train_kl", "eval_kl"]
        + [f"recall@{k}" for k in ks]
        + [
            "lr",
            "seconds",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for entry in history:
            cells = []
            for column in columns:
                value = entry.get(column, "")
                cells.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write(",".join(cells) + "\n")


@app.local_entrypoint()
def main(config: str = "") -> None:
    path = config or os.path.join(THIS_DIR, "config.yaml")
    if not os.path.exists(path):
        fallback = os.path.join(THIS_DIR, "config.example.yaml")
        print(f"warning: {path} not found; using {fallback}")
        path = fallback
    with open(path, encoding="utf-8") as handle:
        config_text = handle.read()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"config: {path}\nrun_id: {run_id}")

    result = train_variant.remote(config_text=config_text, run_id=run_id)
    summary = result["summary"]

    out_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(out_dir, "curves.png"), "wb") as handle:
        handle.write(result["png"])
    _write_history_csv(
        os.path.join(out_dir, "history.csv"),
        summary["history"],
        [int(k) for k in summary["config"]["eval"]["ks"]],
    )

    print(f"\nrow_id: {result['row_id']}")
    print(
        f"train tokens: {summary['n_train_tokens']} | eval tokens: {summary['n_eval_tokens']}"
    )
    print(
        f"best epoch: {summary['best_epoch']} (eval_kl={summary['best_eval_kl']}) | "
        f"final: {summary['final_metrics']}"
    )
    print(f"artifacts mirrored to {out_dir}")
    print(
        "checkpoint on volume: "
        f"modal volume get deepseek-v4-flash-training {run_id}/checkpoint.safetensors"
    )
