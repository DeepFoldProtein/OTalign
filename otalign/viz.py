from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_plan_with_domains(
    P: np.ndarray,
    f: Optional[np.ndarray] = None,
    g: Optional[np.ndarray] = None,
    domains_a: Optional[list[tuple[int, int]]] = None,  # [(start, end), ...] on sequence A
    domains_b: Optional[list[tuple[int, int]]] = None,  # [(start, end), ...] on sequence B
    boxes: Optional[list[tuple[int, int, int, int, float]]] = None,  # from clustering (r0,r1,c0,c1,score)
    title: str = "Transport Plan with Domain Overlays",
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    colorbar: Optional[str] = "transport mass",
    cmap: Any = "viridis",
):
    """
    Visualize the transport plan (P) with optional domain boxes for A and B and discovered boxes.
    Optionally plots bar graphs for potentials f (y-axis) and g (x-axis).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    divider = make_axes_locatable(ax)

    im = ax.imshow(P, origin="upper", aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xlabel("Residues (B)" if label_b is None else label_b)
    ax.set_ylabel("Residues (A)" if label_a is None else label_a)
    ax.set_title(title)

    if g is not None:
        ax_bar_x = divider.append_axes("top", size="20%", pad=0.1, sharex=ax)
        ax_bar_x.bar(np.arange(P.shape[1]), g, color="tab:red")
        ax_bar_x.tick_params(axis="x", labelbottom=False)
        ax_bar_x.spines["top"].set_visible(False)
        ax_bar_x.spines["right"].set_visible(False)
        ax_bar_x.spines["left"].set_visible(False)
        ax_bar_x.set_yticks([])
        ax_bar_x.set_ylabel(r"$g$")

    if f is not None:
        ax_bar_y = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)
        ax_bar_y.barh(np.arange(P.shape[0]), f, color="tab:green")
        ax_bar_y.tick_params(axis="y", labelleft=False)
        ax_bar_y.spines["top"].set_visible(False)
        ax_bar_y.spines["right"].set_visible(False)
        ax_bar_y.spines["bottom"].set_visible(False)
        ax_bar_y.set_xticks([])
        ax_bar_y.set_xlabel(r"$f$")

    if colorbar is not None:
        cax = divider.append_axes("right", size="5%", pad=0.2)
        plt.colorbar(im, cax=cax, label=colorbar)

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
    fig.tight_layout()
    return fig, ax
