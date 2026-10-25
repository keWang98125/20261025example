"""
L4-3 渲染灾前 / 灾后两张同视角对比图
=====================================
输出的不是"好看的动画"，而是可逐项对照的两个状态：
    灾前：车辆/货物全部原位，无水，人员全部正常
    灾后：失稳车辆与实际输运终点，货物散落位置，淹没水面，人员失效标记
"""

import json
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from L4_render.gaussians import build_scene_cloud
from L4_render.splat import project_and_render


def render_pair(t_peak=300.0):
    d = np.load(C.F_SCENE)
    z = d["z"].astype(np.float64)
    lu = d["landuse"]
    run = np.load(C.F_RUN)
    metrics = json.load(open(C.F_METRICS, encoding="utf-8"))

    k = int(np.argmin(np.abs(run["t"] - t_peak)))
    h_peak = run["h"][k]
    print(f"[L4] 用 t={run['t'][k]:.0f}s 的水深场作为灾后淹没形态 "
          f"(hmax={h_peak.max():.2f} m)")

    eye, target = C.CAMERA["eye"], C.CAMERA["target"]

    # ---------------- 灾前（恢复基流水面，便于与灾后同标准对比） ----------------
    t0 = time.time()
    h_base = run["h"][0]
    p, c, s, o = build_scene_cloud(water=True, h_field=h_base, z=z, lu=lu)
    print(f"[L4] 灾前点云: {len(p)} 个高斯")
    img_pre = project_and_render(p, c, s, o, C.IMG_W, C.IMG_H, eye, target, C.FOV_DEG)
    print(f"[L4] 灾前渲染完成 {time.time()-t0:.1f}s")

    # ---------------- 灾后 ----------------
    veh_states = []
    for i, v in enumerate(metrics["vehicles"]):
        veh_states.append((v["x1"], v["y1"], np.deg2rad(15 * i), v["type"], i))
    cargo_states = [(b["x1"], b["y1"], 0) for b in metrics["cargo_boxes"]]
    human_states = [(h["x"], h["y"], 0 if h["level"] == 0 else 1) for h in metrics["humans"]]

    t0 = time.time()
    p, c, s, o = build_scene_cloud(vehicle_states=veh_states, cargo_states=cargo_states,
                                   human_states=human_states, water=True, h_field=h_peak,
                                   z=z, lu=lu)
    print(f"[L4] 灾后点云: {len(p)} 个高斯")
    img_post = project_and_render(p, c, s, o, C.IMG_W, C.IMG_H, eye, target, C.FOV_DEG)
    print(f"[L4] 灾后渲染完成 {time.time()-t0:.1f}s")

    np.savez_compressed(C.OUT / "render.npz", pre=img_pre, post=img_post)
    return img_pre, img_post, h_peak, metrics


def make_figure(img_pre, img_post, h_peak, metrics):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.0), dpi=130)
    ax[0].imshow(img_pre); ax[0].set_title("BEFORE  |  3DGS scene, pre-disaster", fontsize=12)
    ax[1].imshow(img_post); ax[1].set_title("AFTER  |  flood impact, state-level output", fontsize=12)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    nrel = sum(1 for v in metrics["vehicles"] if v["released"])
    txt = (f"vehicles displaced: {nrel}/{len(metrics['vehicles'])}   |   "
           f"cargo boxes drifted: {sum(b['released'] for b in metrics['cargo_boxes'])}/"
           f"{len(metrics['cargo_boxes'])}   |   "
           f"human risk cells: {metrics['human_risk_stats']['affected_cells']} m²")
    fig.text(0.5, 0.02, txt, ha="center", fontsize=9, color="#444")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = C.OUT / "compare_before_after.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[L4] 输出: {out}")
    return out


if __name__ == "__main__":
    a, b, h, m = render_pair()
    make_figure(a, b, h, m)
