"""生成论文与看板所需的全部配图（英文标注，避免中文字体缺失）"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def fig_hydrograph():
    run = np.load(C.F_RUN)
    t = run["t"]
    from L2_physics.local_inertial import hydrograph
    q = np.array([hydrograph(float(x)) for x in np.linspace(0, t[-1], 300)])
    tt = np.linspace(0, t[-1], 300)
    # 溃口出口处（x=110, y=78）的水深与流速过程
    j, i = 78, 110
    h_br = run["h"][:, j, i]
    u_br = np.hypot(run["ux"][:, j, i], run["uy"][:, j, i])

    fig, ax1 = plt.subplots(figsize=(7.2, 3.0))
    ax1.plot(tt, q, color="#1f4e79", lw=1.6, label="inflow unit discharge q(t)")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("q (m²/s)", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1.axvline(C.T_PEAK, ls="--", lw=0.8, color="#999")
    ax1.text(C.T_PEAK + 10, q.max() * 0.85, "flood peak", fontsize=7.5, color="#666")
    ax2 = ax1.twinx()
    ax2.plot(t, h_br, color="#c0392b", lw=1.6, label="depth at breach outlet")
    ax2.set_ylabel("depth (m)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_title("Upstream hydrograph and breach-outlet response", fontsize=10)
    fig.tight_layout(); fig.savefig(C.OUT / "fig_hydrograph.png", dpi=160); plt.close(fig)


def fig_depth_map():
    run = np.load(C.F_RUN)
    z = run["z"]
    hmax = run["h"].max(0)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(np.where(hmax > 0.05, hmax, np.nan), origin="lower",
                   cmap="YlGnBu", extent=[0, C.NX, 0, C.NY], vmin=0, vmax=2.2)
    cb = plt.colorbar(im, ax=ax, shrink=0.85); cb.set_label("max depth (m)")
    ax.contour(z, levels=[z.min() + 0.5], colors="k", linewidths=0.3, alpha=0.25,
               extent=[0, C.NX, 0, C.NY])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Maximum inundation depth over event", fontsize=10)
    fig.tight_layout(); fig.savefig(C.OUT / "fig_depth_map.png", dpi=160); plt.close(fig)


def fig_risk():
    r = np.load(C.OUT / "risk_map.npz")
    risk = r["risk"]
    cmap = ListedColormap(["#f2f2f0", "#f6c445", "#e8743b", "#8e1b1b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(risk, origin="lower", cmap=cmap, norm=norm, extent=[0, C.NX, 0, C.NY])
    ax.set_xlim(40, 180); ax.set_ylim(40, 180)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Human instability risk (Milanesi et al., 2015 criteria)", fontsize=9.5)
    ax.legend(handles=[Patch(fc="#f2f2f0", ec="k", lw=.3, label="safe"),
                       Patch(fc="#f6c445", label="sliding"),
                       Patch(fc="#e8743b", label="toppling"),
                       Patch(fc="#8e1b1b", label="drowning")],
              loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(C.OUT / "fig_risk.png", dpi=160); plt.close(fig)


def fig_vehicles():
    m = json.load(open(C.F_METRICS, encoding="utf-8"))
    run = np.load(C.F_RUN); z = run["z"]; hmax = run["h"].max(0)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(np.where(hmax > 0.05, hmax, np.nan), origin="lower", cmap="YlGnBu",
              extent=[0, C.NX, 0, C.NY], alpha=0.75, vmin=0, vmax=2.2)
    for v in m["vehicles"]:
        col = "#c0392b" if v["released"] else "#2c7a4b"
        tr = np.array(v["traj"])
        if len(tr) > 1:
            ax.plot(tr[:, 1], tr[:, 2], "-", color=col, lw=1.0, alpha=0.85)
        ax.plot(v["x0"], v["y0"], "o", mfc="white", mec=col, ms=5, mew=1.2)
        ax.plot(v["x1"], v["y1"], "s", color=col, ms=5)
        ax.annotate("", xy=(v["x1"], v["y1"]), xytext=(v["x0"], v["y0"]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.8, alpha=0.5,
                                    shrinkA=4, shrinkB=4))
    ax.set_xlim(60, 180); ax.set_ylim(60, 180)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Vehicle trajectories (circle=initial, square=final)", fontsize=9.5)
    ax.legend(handles=[Patch(fc="#c0392b", label="failed / displaced"),
                       Patch(fc="#2c7a4b", label="stable")], fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(C.OUT / "fig_vehicles.png", dpi=160); plt.close(fig)


def fig_framework():
    """四层架构示意图"""
    layers = [
        ("L1  SCENE READINESS", "Gaussian Grouping  ->  InsertAny3D / TRACE",
         "static colliders + movable rigid assets + semantic masks", "#dbe9f6"),
        ("L2  PHYSICS", "Local-Inertial SWE (this repo)  /  RAF-Genesis  /  DualSPHysics",
         "depth & velocity fields + rigid-body trajectories", "#d8ead8"),
        ("L3  ASSESSMENT", "vehicle buoyancy-drag  |  cargo RMS spread  |  Milanesi human model",
         "failure labels + transport endpoints + risk raster", "#f6e3d0"),
        ("L4  COMPARISON", "3DGS rasterizer (this repo)  /  gsplat",
         "before-after state images, same viewpoint", "#eee0ee"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axis("off")
    y = 0.86
    for name, mid, out, col in layers:
        ax.add_patch(plt.Rectangle((0.05, y - 0.145), 0.90, 0.135,
                                   facecolor=col, edgecolor="#5b6b7a", lw=0.8))
        ax.text(0.09, y - 0.035, name, fontsize=9.5, fontweight="bold", va="top")
        ax.text(0.09, y - 0.075, mid, fontsize=7.4, va="top", color="#20303f")
        ax.text(0.09, y - 0.108, "-> " + out, fontsize=7.0, va="top", color="#5a5a5a")
        y -= 0.175
        if y > 0.14:
            ax.annotate("", xy=(0.50, y + 0.028), xytext=(0.50, y + 0.052),
                        arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0))
    ax.text(0.5, 0.03, "fluent input: rainfall / water-level boundary  ->  one-way coupling",
            ha="center", fontsize=7.2, color="#777", style="italic")
    ax.set_title("Four-layer decoupled architecture", fontsize=10.5, pad=6)
    fig.tight_layout(); fig.savefig(C.OUT / "fig_framework.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    fig_framework(); print("[fig] framework")
    fig_hydrograph(); print("[fig] hydrograph")
    fig_depth_map(); print("[fig] depth map")
    fig_risk(); print("[fig] risk")
    fig_vehicles(); print("[fig] vehicles")
    print("done")
