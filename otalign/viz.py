from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_plan_with_domains(
    P: np.ndarray,
    domains_a: Optional[list[tuple[int, int]]] = None,  # [(start, end), ...] on sequence A
    domains_b: Optional[list[tuple[int, int]]] = None,  # [(start, end), ...] on sequence B
    boxes: Optional[list[tuple[int, int, int, int, float]]] = None,  # from clustering (r0,r1,c0,c1,score)
    title: str = "Transport Plan with Domain Overlays",
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
):
    """
    Visualize the transport plan (P) with optional domain boxes for A and B and discovered boxes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(P, origin="upper", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="transport mass")
    ax.set_xlabel("Residues (B)" if label_b is None else label_b)
    ax.set_ylabel("Residues (A)" if label_a is None else label_a)
    ax.set_title(title)
    # overlay known domain ranges (A = rows; B = cols)
    if domains_a is not None:
        for s, e in domains_a:
            ax.hlines([s, e], xmin=-0.5, xmax=P.shape[1] - 0.5, colors="w", linestyles="--", linewidth=0.7, alpha=0.8)
    if domains_b is not None:
        for s, e in domains_b:
            ax.vlines([s, e], ymin=-0.5, ymax=P.shape[0] - 0.5, colors="w", linestyles="--", linewidth=0.7, alpha=0.8)
    # overlay discovered boxes
    if boxes is not None:
        for r0, r1, c0, c1, sc in boxes:
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), (c1 - c0 + 1), (r1 - r0 + 1), fill=False, edgecolor="cyan", linewidth=1.2))
            ax.text(c0, r0 - 1, f"{sc:.2f}", color="cyan", fontsize=8)
    plt.tight_layout()
    return fig, ax
