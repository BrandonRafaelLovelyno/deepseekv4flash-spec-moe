"""DeepSeek-V4-Flash weight share -- parameter accounting + a hand-drawn pie.

Splits the model's parameters three ways for a single token:

* unused routed experts  -- 250 of the 256 experts in each of the 43 MoE layers
* activated experts      -- the 6 experts routed to per layer, per token
* other weights          -- attention, the always-on shared expert, embeddings,
                            head, MTP block, gates, norms

Tensor shapes come from the reference architecture config
(``checkpoints/inference/config.json``); the resulting fp4/fp8 byte estimate is
cross-checked against the safetensors index metadata.

The chart is styled after the project's Excalidraw diagrams: Comic Neue,
paper background, pastel yellow / blue / green fills, ink outlines.

Usage:
    python chart.py
"""

from __future__ import annotations

import io
import json
import os
import urllib.request

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO_ROOT, "checkpoints", "inference", "config.json")
INDEX_PATH = os.path.join(REPO_ROOT, "checkpoints", "model.safetensors.index.json")
FONT_DIR = os.path.join(REPO_ROOT, "fonts")
OUT_PATH = os.path.join(REPO_ROOT, "chart.png")

FONT_URLS = {
    "ComicNeue-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Regular.ttf"
    ),
    "ComicNeue-Bold.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/comicneue/ComicNeue-Bold.ttf"
    ),
}

# Palette lifted from the reference diagram.
PAPER = "#FFFFFF"
INK = "#2B2B2B"
COLORS = {"unused": "#F4DE9A", "activated": "#6FA8DC", "other": "#AED9A5"}
EXPLODE = {"unused": 0.0, "activated": 0.08, "other": 0.0}

# fp4 experts: 0.5 byte/param + one fp8 scale per 32 elements along K.
FP4_BYTES_PER_PARAM = 0.5 * (1.0 + 1.0 / 16.0)
FP8_BYTES_PER_PARAM = 1.0


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _compressor_params(dim: int, head_dim: int, ratio: int) -> int:
    coff = 1 + (1 if ratio == 4 else 0)
    return ratio * coff * head_dim + 2 * (dim * coff * head_dim) + head_dim


def compute_share(cfg: dict) -> dict:
    """Per-category parameter counts plus the three-way split."""
    vocab, dim, inter, n_layers = (
        cfg["vocab_size"], cfg["dim"], cfg["moe_inter_dim"], cfg["n_layers"],
    )
    n_experts, k, n_shared = (
        cfg["n_routed_experts"], cfg["n_activated_experts"], cfg["n_shared_experts"],
    )
    n_heads, head_dim, q_lora = cfg["n_heads"], cfg["head_dim"], cfg["q_lora_rank"]
    o_groups, o_lora = cfg["o_groups"], cfg["o_lora_rank"]
    idx_heads, idx_head_dim = cfg["index_n_heads"], cfg["index_head_dim"]
    hc, ratios, n_hash = cfg["hc_mult"], cfg["compress_ratios"], cfg["n_hash_layers"]
    n_targets = len(cfg["dspark_target_layer_ids"])

    expert = 3 * dim * inter  # w1, w2, w3 per expert

    attn = (
        dim * q_lora
        + q_lora * n_heads * head_dim
        + dim * head_dim
        + (n_heads * head_dim // o_groups) * (o_groups * o_lora)
        + (o_groups * o_lora) * dim
        + q_lora + head_dim + n_heads
    )
    compressor = sum(_compressor_params(dim, head_dim, ratios[i]) for i in range(n_layers))
    indexer = sum(
        q_lora * idx_heads * idx_head_dim
        + dim * idx_heads
        + _compressor_params(dim, idx_head_dim, 4)
        for i in range(n_layers)
        if ratios[i] == 4
    )
    gate = n_experts * dim * n_layers + n_experts * (n_layers - n_hash) + vocab * k * n_hash
    mix = (2 + hc) * hc
    hc_dim = hc * dim
    hyper = (2 * mix * hc_dim + 2 * mix + 6) * n_layers
    norms = 2 * dim * n_layers
    globals_ = vocab * dim + vocab * dim + dim + (hc * hc_dim + hc + 1)

    # MTP block: shipped in the checkpoint but excluded from DeepSeek's 284B
    # headline (and from the 13B active count).
    mtp_experts = (n_experts + n_shared) * expert
    mtp_other = (
        attn + n_experts * dim + n_experts
        + hyper // n_layers + norms // n_layers
        + dim * n_targets * dim + dim * dim
    )
    mtp = mtp_experts + mtp_other

    experts_routed = n_experts * n_layers * expert
    experts_active = k * n_layers * expert
    shared = n_shared * n_layers * expert
    attn_total = attn * n_layers

    # Weights touched by one token's forward pass: the k routed experts + shared
    # expert + attention/compressor/indexer/gate/hc/norms. This is DeepSeek's
    # "13B activated" -- it DOES include the used experts. Embeddings/head are a
    # lookup (not dense per-token compute) and MTP is not run, so both are out.
    active = (
        experts_active + shared + attn_total
        + compressor + indexer + gate + hyper + norms
    )
    # Everything that is not a routed expert, in the base model.
    other_base = (
        shared + attn_total + compressor + indexer + gate + hyper + norms + globals_
    )
    base_total = experts_routed + other_base  # DeepSeek's 284B
    checkpoint_total = base_total + mtp  # all tensors shipped on disk

    bytes_ = (
        (experts_routed + shared + mtp_experts) * FP4_BYTES_PER_PARAM
        + (attn_total + compressor + indexer + gate + hyper + norms + globals_ + mtp_other)
        * FP8_BYTES_PER_PARAM
    )
    return {
        "base_total": base_total,
        "checkpoint_total": checkpoint_total,
        "active": active,
        "mtp": mtp,
        "expert_params": expert,
        "experts_routed": experts_routed,
        "experts_active": experts_active,
        "other": other_base,
        "bytes": bytes_,
        "slices": [
            {
                "key": "unused",
                "label": "Unused routed experts",
                "detail": f"{(n_experts - k) * n_layers} experts",
                "params": experts_routed - experts_active,
            },
            {
                "key": "activated",
                "label": "Activated experts / token",
                "detail": f"{k * n_layers} experts",
                "params": experts_active,
            },
            {
                "key": "other",
                "label": "Other weights",
                "detail": "attention, shared expert, embeddings…",
                "params": other_base,
            },
        ],
    }


def _ensure_font() -> str:
    """Return a Comic Neue family name, downloading the TTFs on first use."""
    import matplotlib.font_manager as fm

    os.makedirs(FONT_DIR, exist_ok=True)
    family = None
    for name, url in FONT_URLS.items():
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
            except OSError as exc:  # offline: fall back gracefully
                print(f"warning: could not fetch {name}: {exc}")
                continue
        fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=path).get_name()
    return family or "DejaVu Sans"


def render_pie(share: dict) -> bytes:
    """Render the three-way split as a hand-drawn-style pie (PNG bytes)."""
    import matplotlib.pyplot as plt

    style = {
        "font.family": _ensure_font(),
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "path.sketch": (1.0, 100, 2),  # subtle hand-drawn wobble
    }
    with plt.rc_context(style):
        slices, total = share["slices"], share["base_total"]
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        wedges, _, autotexts = ax.pie(
            [s["params"] for s in slices],
            colors=[COLORS[s["key"]] for s in slices],
            explode=[EXPLODE[s["key"]] for s in slices],
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": INK, "linewidth": 2.4},
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
            pctdistance=0.74,
            textprops={"color": INK, "fontsize": 17, "fontweight": "bold"},
        )
        for text in autotexts:
            text.set_color(INK)

        active = next(s for s in slices if s["key"] == "activated")

        labels = [
            f"{s['label']}  ({s['detail']})\n"
            f"{s['params'] / 1e9:.1f} B  ·  {100 * s['params'] / total:.1f}%"
            for s in slices
        ]
        legend = ax.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=15,
            handlelength=1.0,
            handleheight=1.0,
            labelspacing=0.85,
        )
        for text in legend.get_texts():
            text.set_color(INK)

        ax.set_title(
            f"DeepSeek-V4-Flash weight share\n"
            f"~{total / 1e9:.0f} B total  ·  ~{share['active'] / 1e9:.0f} B active / token",
            fontsize=20,
            fontweight="bold",
            color=INK,
            pad=8,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-1.08, 1.15)
        ax.set_ylim(-1.08, 1.15)
        ax.set_axis_off()
        ax.set_anchor("W")
        always_on = (share["active"] - active["params"]) / 1e9
        fig.text(
            0.5,
            0.005,
            f"Active {share['active'] / 1e9:.1f} B = {active['params'] / 1e9:.2f} B "
            f"experts routed to + {always_on:.1f} B always-on layer weights. "
            f"Embeddings/head (~{(share['other'] - share['active'] + active['params']) / 1e9:.1f} B, "
            f"a lookup) and a shipped {share['mtp'] / 1e9:.1f} B MTP head are excluded.",
            ha="center",
            va="bottom",
            fontsize=12,
            color=INK,
        )

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return buffer.getvalue()


def main() -> None:
    cfg = load_config()
    share = compute_share(cfg)
    with open(OUT_PATH, "wb") as handle:
        handle.write(render_pie(share))

    total = share["base_total"]
    print(f"expert params         : {share['expert_params'] / 1e6:.2f} M")
    print(f"base total            : {total / 1e9:.2f} B (DeepSeek's 284B, no MTP)")
    print(f"active / token        : {share['active'] / 1e9:.2f} B (DeepSeek's 13B)")
    print(f"  + MTP head shipped  : {share['mtp'] / 1e9:.2f} B")
    print(f"checkpoint total      : {share['checkpoint_total'] / 1e9:.2f} B")
    for s in share["slices"]:
        print(
            f"{s['label']:<22}: {s['params'] / 1e9:8.2f} B "
            f"({100 * s['params'] / total:4.1f}%)  [{s['detail']}]"
        )
    with open(INDEX_PATH, encoding="utf-8") as handle:
        checkpoint = json.load(handle)["metadata"]["total_size"]
    print(f"reconstructed size    : {share['bytes'] / 1e9:7.1f} GB (fp4 experts, fp8 rest)")
    print(f"checkpoint index size : {checkpoint / 1e9:7.1f} GB")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
