"""
Render all curriculum levels as images for report.
Each level: side-by-side floor maps with color-coded tiles, legend, and info.

Usage: python render_maps.py
"""
import copy
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from mfm.env import MazeEnv, CURRICULUM, Tile

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "level_maps")
os.makedirs(OUT_DIR, exist_ok=True)

TILE_COLORS = {
    int(Tile.WALL):        (0.15, 0.15, 0.20),
    int(Tile.FLOOR):       (0.92, 0.90, 0.85),
    int(Tile.STAIR_UP):    (0.40, 0.75, 0.95),
    int(Tile.STAIR_DOWN):  (0.20, 0.50, 0.80),
    int(Tile.GOAL):        (1.00, 0.84, 0.00),
    int(Tile.TRAP):        (0.90, 0.30, 0.30),
    int(Tile.START):       (0.30, 0.85, 0.40),
    int(Tile.DOOR):        (0.60, 0.35, 0.15),
    int(Tile.KEY):         (1.00, 0.65, 0.00),
    int(Tile.HEALTH_PACK): (0.95, 0.45, 0.55),
    int(Tile.AMMO_PACK):   (0.55, 0.55, 0.55),
}

LEGEND_ITEMS = [
    ("Wall",        TILE_COLORS[int(Tile.WALL)]),
    ("Floor",       TILE_COLORS[int(Tile.FLOOR)]),
    ("Start",       TILE_COLORS[int(Tile.START)]),
    ("Goal",        TILE_COLORS[int(Tile.GOAL)]),
    ("Stair Up",    TILE_COLORS[int(Tile.STAIR_UP)]),
    ("Stair Down",  TILE_COLORS[int(Tile.STAIR_DOWN)]),
    ("Trap",        TILE_COLORS[int(Tile.TRAP)]),
    ("Door",        TILE_COLORS[int(Tile.DOOR)]),
    ("Key",         TILE_COLORS[int(Tile.KEY)]),
    ("Health Pack", TILE_COLORS[int(Tile.HEALTH_PACK)]),
    ("Ammo Pack",   TILE_COLORS[int(Tile.AMMO_PACK)]),
]

ENEMY_MARKERS = {
    "patrol": {"marker": "^", "color": "#E91E63", "label": "Patrol"},
    "chaser": {"marker": "s", "color": "#9C27B0", "label": "Chaser"},
    "sniper": {"marker": "D", "color": "#F44336", "label": "Sniper"},
}


def grid_to_rgb(grid):
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            t = int(grid[y][x])
            img[y, x] = TILE_COLORS.get(t, (0.5, 0.5, 0.5))
    return img


def render_level(level_id, config, seed=42, n_samples=3):
    """Render n_samples map instances for a level."""
    for sample_idx in range(n_samples):
        s = seed + sample_idx * 100
        env = MazeEnv(config=copy.deepcopy(config))
        env.reset(seed=s)

        nf = env.nf
        floors = env.floors
        enemies = env.enemies
        start = (env.agent.x, env.agent.y, env.agent.floor)
        goal = env.goal
        w, h = env.mw, env.mh

        fig_w = max(5, 4.5 * nf + 1.5)
        fig, axes = plt.subplots(1, nf, figsize=(fig_w, 5.5), squeeze=False)

        level_name = config.get("name", f"Level {level_id}")
        n_enemies = len(config.get("enemies", []))
        n_locks = len(config.get("locks", []))
        fig.suptitle(
            f"Level {level_id}: {level_name}  —  "
            f"{w}x{h}, {nf} floor{'s' if nf > 1 else ''}, "
            f"{n_enemies} enem{'ies' if n_enemies != 1 else 'y'}, "
            f"{n_locks} lock{'s' if n_locks != 1 else ''}  "
            f"(seed={s})",
            fontsize=13, fontweight="bold", y=0.98,
        )

        for f_idx in range(nf):
            ax = axes[0][f_idx]
            grid = floors[f_idx]
            img = grid_to_rgb(grid)
            ax.imshow(img, interpolation="nearest", origin="upper")

            sx, sy, sf = start
            if sf == f_idx:
                ax.plot(sx, sy, marker="o", color="#00E676", markersize=12,
                        markeredgecolor="black", markeredgewidth=1.5, zorder=10)
                ax.annotate("S", (sx, sy), fontsize=7, fontweight="bold",
                            ha="center", va="center", color="black", zorder=11)

            gx, gy, gf = goal
            if gf == f_idx:
                ax.plot(gx, gy, marker="*", color="#FFD600", markersize=16,
                        markeredgecolor="black", markeredgewidth=1.2, zorder=10)
                ax.annotate("G", (gx, gy), fontsize=6, fontweight="bold",
                            ha="center", va="center", color="black", zorder=11)

            for e in enemies:
                if e.floor == f_idx:
                    style = ENEMY_MARKERS.get(e.etype, ENEMY_MARKERS["patrol"])
                    ax.plot(e.x, e.y, marker=style["marker"],
                            color=style["color"], markersize=10,
                            markeredgecolor="white", markeredgewidth=1.0,
                            zorder=9)

            ax.set_title(f"Floor {f_idx}", fontsize=11, pad=4)
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_xticks(range(0, w, max(1, w // 5)))
            ax.set_yticks(range(0, h, max(1, h // 5)))
            ax.tick_params(labelsize=7)
            ax.set_aspect("equal")

            for spine in ax.spines.values():
                spine.set_color("#888")

        legend_handles = [
            mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=n)
            for n, c in LEGEND_ITEMS
            if _tile_present(floors, n)
        ]
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#00E676",
                       markeredgecolor="black", markersize=8, label="Agent Start")
        )
        legend_handles.append(
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD600",
                       markeredgecolor="black", markersize=10, label="Goal")
        )
        for etype, style in ENEMY_MARKERS.items():
            if any(e.etype == etype for e in enemies):
                legend_handles.append(
                    plt.Line2D([0], [0], marker=style["marker"], color="w",
                               markerfacecolor=style["color"],
                               markeredgecolor="white", markersize=8,
                               label=style["label"])
                )

        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(6, len(legend_handles)), fontsize=8,
                   framealpha=0.9, edgecolor="#ccc", borderpad=0.6)

        plt.tight_layout(rect=[0, 0.08, 1, 0.94])
        fname = f"level_{level_id}_{level_name.lower()}_seed{s}.png"
        path = os.path.join(OUT_DIR, fname)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fname}")


def _tile_present(floors, tile_name):
    name_to_tile = {
        "Wall": Tile.WALL, "Floor": Tile.FLOOR, "Start": Tile.START,
        "Goal": Tile.GOAL, "Stair Up": Tile.STAIR_UP,
        "Stair Down": Tile.STAIR_DOWN, "Trap": Tile.TRAP,
        "Door": Tile.DOOR, "Key": Tile.KEY,
        "Health Pack": Tile.HEALTH_PACK, "Ammo Pack": Tile.AMMO_PACK,
    }
    t = name_to_tile.get(tile_name)
    if t is None:
        return False
    for g in floors:
        if np.any(g == int(t)):
            return True
    return False


def main():
    print(f"Rendering maps to: {OUT_DIR}\n")
    for level_id in sorted(CURRICULUM.keys()):
        cfg = CURRICULUM[level_id]
        print(f"Level {level_id}: {cfg['name']} "
              f"({cfg['width']}x{cfg['height']}, {cfg['n_floors']} floors)")
        render_level(level_id, cfg, seed=42, n_samples=3)
        print()
    print(f"\nDone! All maps in: {OUT_DIR}")


if __name__ == "__main__":
    main()
