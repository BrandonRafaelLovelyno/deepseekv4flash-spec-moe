"""Matplotlib figures for a training run.

matplotlib is imported inside each function, so importing this module (from a
test or the local entrypoint) never pulls the plotting stack. Each function
returns PNG bytes; writing them to disk is the caller's concern.
"""

from __future__ import annotations

READY_COLORS = {0.25: "#4C72B0", 0.5: "#DD8452", 0.75: "#55A868"}
LINESTYLES = ["-", "--", "-.", ":"]


def _eval_epochs(history: list[dict]) -> list[int]:
    return [h["epoch"] for h in history if "eval_kl" in h]


def training_curves_png(history: list[dict], ks: list[int], row_id: str) -> bytes:
    """Loss (train vs eval) and recall@k curves, side by side, as PNG bytes."""
    import io

    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_recall) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax_loss.plot(
        [h["epoch"] for h in history], [h["train_kl"] for h in history], label="train"
    )
    eval_epochs = _eval_epochs(history)
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
    return buffer.getvalue()


def ready_recall_png(
    history: list[dict], ks: list[int], ratios: tuple[float, ...], row_id: str
) -> bytes:
    """Ready recall: one line per (ratio, k), plus the resident-only baseline."""
    import io

    import matplotlib.pyplot as plt

    eval_epochs = _eval_epochs(history)
    fig_ready, ax_ready = plt.subplots(figsize=(6.5, 4.0))
    for r in ratios:
        tag = f"r{int(round(r * 100))}"
        color = READY_COLORS.get(r)
        for k, ls in zip(ks, LINESTYLES):
            key = f"ready_recall_{tag}@{k}"
            ax_ready.plot(
                eval_epochs,
                [h[key] for h in history if key in h],
                ls,
                markersize=3,
                color=color,
                label=f"ready {int(round(r * 100))}% @{k}",
            )
        baseline = f"resident_recall_{tag}"
        ax_ready.plot(
            eval_epochs,
            [h[baseline] for h in history if baseline in h],
            color=color,
            linewidth=1.0,
            alpha=0.4,
            linestyle=(0, (1, 3)),
            label=f"resident {int(round(r * 100))}%",
        )
    ax_ready.set_xlabel("epoch")
    ax_ready.set_ylabel("recall")
    ax_ready.set_ylim(0.0, 1.0)
    ax_ready.set_title(f"ready recall: {row_id}")
    ax_ready.legend(fontsize=7, ncol=2)
    ax_ready.grid(alpha=0.3)
    fig_ready.tight_layout()
    buffer_ready = io.BytesIO()
    fig_ready.savefig(buffer_ready, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig_ready)
    return buffer_ready.getvalue()
