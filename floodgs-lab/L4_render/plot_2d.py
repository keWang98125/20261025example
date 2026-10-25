"""
L4-5 二维俯视特写动画
=====================
3D 渲染在近景下会糊（地形高斯按 2 m 采样，尺度不匹配）。
近景改用二维俯视：水深填色 + 流速矢量 + 物体按真实位姿绘制，
分辨率高、信息密度大，也更接近论文里常见的淹没图。

用法：
    python L4_render/plot_2d.py --tend 500 --dt 4 --fps 15
"""

import argparse
import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7})


def interp_traj(traj, t):
    if len(traj) < 2:
        return traj[0][1], traj[0][2]
    a = np.array(traj, dtype=float)
    ts, xs, ys = a[:, 0], a[:, 1], a[:, 2]
    if t <= ts[0]:
        return xs[0], ys[0]
    if t >= ts[-1]:
        return xs[-1], ys[-1]
    return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))


def heading_at(traj, t):
    x0, y0 = interp_traj(traj, t - 4.0)
    x1, y1 = interp_traj(traj, t + 4.0)
    if abs(x1 - x0) + abs(y1 - y0) < 0.05:
        return None
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def field_at(arr, t, T):
    k = int(np.searchsorted(T, t))
    if k <= 0:
        return arr[0]
    if k >= len(T):
        return arr[-1]
    a = (t - T[k - 1]) / max(T[k] - T[k - 1], 1e-9)
    return (1 - a) * arr[k - 1] + a * arr[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tend", type=float, default=500.0)
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--xlim", default="58,172")
    ap.add_argument("--ylim", default="62,148")
    ap.add_argument("--out", default="outputs/flood_impact_2d.mp4")
    args = ap.parse_args()

    x0, x1 = [float(v) for v in args.xlim.split(",")]
    y0, y1 = [float(v) for v in args.ylim.split(",")]

    d = np.load(C.F_SCENE); z = d["z"]; lu = d["landuse"]
    run = np.load(C.F_RUN)
    T = run["t"].astype(np.float64)
    Hseq, UX, UY = run["h"], run["ux"], run["uy"]
    m = json.load(open(C.F_METRICS, encoding="utf-8"))

    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(11.2, 6.0), dpi=110)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.022], wspace=0.02,
                          left=0.05, right=0.975, top=0.94, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    # colorbar 只建一次（放在独立 cax 上），否则每帧 clear 会不断堆叠
    _sm = plt.cm.ScalarMappable(norm=Normalize(0, 1.05), cmap="Blues")
    _sm.set_array([])
    _cb = fig.colorbar(_sm, cax=cax)
    _cb.set_label("depth (m)", fontsize=8)

    times = np.arange(0, args.tend + 1e-6, args.dt)
    frames = []
    t_start = time.time()
    total_rel = sum(1 for v in m["vehicles"] if v["released"])

    # 静态底图：地形
    extent = [0, C.NX, 0, C.NY]
    msh = None
    for n, t in enumerate(times):
        ax.clear()
        h = field_at(Hseq, t, T).astype(float)
        ux = field_at(UX, t, T).astype(float)
        uy = field_at(UY, t, T).astype(float)

        # 地形灰度底
        ax.imshow(z, origin="lower", cmap="Greys", extent=extent,
                  alpha=0.35, vmin=z.min() - 3, vmax=z.max() + 3)
        # 水深
        hm = np.where(h > 0.04, h, np.nan)
        im = ax.imshow(hm, origin="lower", cmap="Blues", extent=extent,
                       vmin=0, vmax=1.05, alpha=0.95)
        # 流速矢量（稀疏）
        st = 9
        jj, ii = np.mgrid[0:C.NY:st, 0:C.NX:st]
        hh = h[jj, ii]
        keep = hh > 0.08
        if keep.any():
            ax.quiver((ii + 0.5)[keep], (jj + 0.5)[keep],
                      ux[jj, ii][keep], uy[jj, ii][keep],
                      color="#0d3a58", alpha=0.80, scale=42, width=0.0038,
                      headwidth=4.2, headlength=4.8)

        # 货物
        for b in m["cargo_boxes"]:
            bx, by = interp_traj(b["traj"], t)
            rel = b["released"] and (b["t_release"] is not None) and t >= b["t_release"]
            ax.add_patch(Rectangle((bx - 0.4, by - 0.4), 0.8, 0.8,
                                   facecolor="#c98f2e" if rel else "#8a6b33",
                                   edgecolor="white", lw=0.3, alpha=0.95, zorder=5))

        # 车辆
        for i, v in enumerate(m["vehicles"]):
            vx, vy = interp_traj(v["traj"], t)
            ang = heading_at(v["traj"], t)
            deg = 15 * i if ang is None else ang
            p = C.VEHICLE_TYPES[v["type"]]
            rel = v["released"] and (v["t_release"] is not None) and t >= v["t_release"]
            fc = "#d93a2b" if rel else "#2f6fb5"
            r = Rectangle((vx - p["L"] / 2, vy - p["W"] / 2), p["L"], p["W"],
                          angle=deg, rotation_point="center",
                          facecolor=fc, edgecolor="white", lw=0.6, zorder=7)
            ax.add_patch(r)
            if rel:
                ax.add_patch(Circle((vx, vy), p["L"] * 0.95, fill=False,
                                    ec="#d93a2b", lw=1.0, alpha=0.45, zorder=6))

        # 人员
        for hm in m["humans"]:
            c = "#e8332a" if hm["level"] == 0 else "#f08a24"
            ax.add_patch(Circle((hm["x"], hm["y"]), 0.85, facecolor=c,
                                edgecolor="white", lw=0.5, zorder=8))

        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

        nrel = sum(1 for v in m["vehicles"]
                   if v["released"] and v["t_release"] is not None and t >= v["t_release"])
        ax.set_title(f"Flood impact  |  t = {t:4.0f} s   "
                     f"max depth {np.nanmax(h):.2f} m   "
                     f"vehicles failed {nrel}/{total_rel}",
                     fontsize=11, loc="left")
        # 图例
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([], [], marker="s", ls="", mfc="#2f6fb5", mec="w", ms=9, label="vehicle stable"),
            Line2D([], [], marker="s", ls="", mfc="#d93a2b", mec="w", ms=9, label="vehicle failed"),
            Line2D([], [], marker="s", ls="", mfc="#c98f2e", mec="w", ms=8, label="cargo (afloat)"),
            Line2D([], [], marker="o", ls="", mfc="#e8332a", mec="w", ms=8, label="person safe"),
            Line2D([], [], marker="o", ls="", mfc="#f08a24", mec="w", ms=8, label="person at risk"),
        ], loc="lower left", fontsize=6.6, ncol=5, framealpha=0.88, handletextpad=0.4,
            columnspacing=0.9, borderpad=0.4)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

        if n % 10 == 0 or n == len(times) - 1:
            el = time.time() - t_start
            print(f"  帧 {n+1}/{len(times)}  t={t:5.0f}s  已用 {el:4.0f}s")

    plt.close(fig)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as iio
    iio.mimsave(out, frames, fps=args.fps, quality=8, macro_block_size=None, codec="libx264")
    print(f"[2d] 输出: {out} ({out.stat().st_size/1e6:.1f} MB, {len(frames)} 帧)")


if __name__ == "__main__":
    main()
