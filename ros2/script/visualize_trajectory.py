import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib\.projections")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import numpy as np
from matplotlib.lines import Line2D


def read_trajectory(csv_path: Path):
    t, x1, y1, x2, y2, x3, y3 = [], [], [], [], [], [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        # Expected headers at minimum:
        # time_s,x1,y1,x2,y2,x3,y3,...
        for row in reader:
            t.append(float(row["time_s"]))
            x1.append(float(row["x1"]))
            y1.append(float(row["y1"]))
            x2.append(float(row["x2"]))
            y2.append(float(row["y2"]))
            x3.append(float(row["x3"]))
            y3.append(float(row["y3"]))
    return t, (x1, y1), (x2, y2), (x3, y3)


def save_static_plot(csv_path: Path, out_png: Path, title: str = "Agent Trajectories"):
    t, (x1, y1), (x2, y2), (x3, y3) = read_trajectory(csv_path)

    # Bounds with margins
    all_x = x1 + x2 + x3
    all_y = y1 + y2 + y3
    if len(all_x) == 0:
        raise RuntimeError("No trajectory points found in CSV.")
    xmin, xmax = min(all_x) - 0.5, max(all_x) + 0.5
    ymin, ymax = min(all_y) - 0.5, max(all_y) + 0.5

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)

    def draw_fading_line(xs, ys, rgb, lw=2.0, min_alpha=0.2, max_alpha=1.0):
        if len(xs) < 2:
            return
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # [N-1, 2, 2]
        nseg = segs.shape[0]
        alphas = np.linspace(min_alpha, max_alpha, nseg, dtype=float)
        colors = np.tile(np.array(list(rgb) + [1.0])[None, :], (nseg, 1))
        colors[:, 3] = alphas
        lc = LineCollection(segs, colors=colors, linewidths=lw, capstyle="round", joinstyle="round")
        ax.add_collection(lc)

    # Draw with increasing opacity along time
    draw_fading_line(x1, y1, rgb=(1.0, 0.0, 0.0), lw=2.0)  # red
    draw_fading_line(x2, y2, rgb=(0.0, 0.6, 0.0), lw=2.0)  # green
    draw_fading_line(x3, y3, rgb=(0.0, 0.0, 1.0), lw=2.0)  # blue

    # Mark starting positions
    ax.plot([x1[0]], [y1[0]], "r^", ms=6, mfc="none", mew=1.5)
    ax.plot([x2[0]], [y2[0]], "g^", ms=6, mfc="none", mew=1.5)
    ax.plot([x3[0]], [y3[0]], "b^", ms=6, mfc="none", mew=1.5)

    # Mark final positions
    ax.plot([x1[-1]], [y1[-1]], "ro", ms=5)
    ax.plot([x2[-1]], [y2[-1]], "go", ms=5)
    ax.plot([x3[-1]], [y3[-1]], "bo", ms=5)

    # Target goal at origin
    ax.plot([0.0], [0.0], "k*", ms=10, label="Target Goal")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="r", lw=2, label="Pursuer"),
        Line2D([0], [0], color="g", lw=2, label="Guard"),
        Line2D([0], [0], color="b", lw=2, label="Target"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="none", markeredgecolor="k",
               markersize=6, linestyle="None", label="Initial Positions"),
        Line2D([0], [0], marker="*", color="k", markersize=10, linestyle="None", label="Target's Goal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gif(csv_path: Path, out_gif: Path, fps: int = 10):
    t, (x1, y1), (x2, y2), (x3, y3) = read_trajectory(csv_path)

    all_x = x1 + x2 + x3
    all_y = y1 + y2 + y3
    if len(all_x) == 0:
        raise RuntimeError("No trajectory points found in CSV.")
    xmin, xmax = min(all_x) - 0.5, max(all_x) + 0.5
    ymin, ymax = min(all_y) - 0.5, max(all_y) + 0.5

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Agent Trajectories (Animated)")

    line1, = ax.plot([], [], "r-", lw=2, label="Pursuer")
    line2, = ax.plot([], [], "g-", lw=2, label="Guard")
    line3, = ax.plot([], [], "b-", lw=2, label="Target")
    pt1, = ax.plot([], [], "ro", ms=5)
    pt2, = ax.plot([], [], "go", ms=5)
    pt3, = ax.plot([], [], "bo", ms=5)

    # Static markers: starts and goal at origin
    # Draw once so they persist for all frames
    # Start markers
    ax.plot([x1[0]], [y1[0]], "r^", ms=6, mfc="none", mew=1.5)
    ax.plot([x2[0]], [y2[0]], "g^", ms=6, mfc="none", mew=1.5)
    ax.plot([x3[0]], [y3[0]], "b^", ms=6, mfc="none", mew=1.5)
    # Goal marker
    ax.plot([0.0], [0.0], "k*", ms=10)

    ax.legend(loc="upper right")

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        pt1.set_data([], [])
        pt2.set_data([], [])
        pt3.set_data([], [])
        return line1, line2, line3, pt1, pt2, pt3

    def animate(i):
        line1.set_data(x1[:i + 1], y1[:i + 1])
        line2.set_data(x2[:i + 1], y2[:i + 1])
        line3.set_data(x3[:i + 1], y3[:i + 1])
        pt1.set_data([x1[i]], [y1[i]])
        pt2.set_data([x2[i]], [y2[i]])
        pt3.set_data([x3[i]], [y3[i]])
        return line1, line2, line3, pt1, pt2, pt3

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(t), interval=1000 // max(fps, 1), blit=True
    )
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_gif), writer="pillow", fps=fps)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectories from trajectory_log.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "trajectory_log.csv",
        help="Path to trajectory_log.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "trajectory_final.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Optional output GIF path (if provided, an animated GIF is also saved)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for GIF animation (if --gif is provided)",
    )
    args = parser.parse_args()

    save_static_plot(args.csv, args.out)
    if args.gif is not None:
        save_gif(args.csv, args.gif, fps=args.fps)


if __name__ == "__main__":
    main()


