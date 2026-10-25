"""
L4-4 动态序列渲染 —— 洪水冲击过程动画
======================================
把 L2 的流场时序 + L3 的物体轨迹，逐帧重建成 3D 高斯场景并渲染，输出 MP4。

关键工程点：
  1. 地形/建筑点云只构建一次（占总量 70%），每帧只重建车辆/货物/人员/水面，
     把单帧成本从 ~1.4 s 压到 ~1.1 s。
  2. 物体位置从 metrics.json 的轨迹按时间线性插值，不重跑物理。
  3. 车辆朝向取轨迹切线方向，失稳车辆用红色高亮。

用法：
    python L4_render/animate.py                # 0~600 s, 5 s/帧 -> MP4
    python L4_render/animate.py --dt 3 --tend 700 --fps 15
"""

import argparse
import json
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from L4_render.gaussians import (terrain_gaussians, box_gaussians, water_gaussians,
                                 PALETTE, VEHICLE_COLORS)
from L4_render.splat import project_and_render


# ---------------------------------------------------------------- 工具
def interp_traj(traj, t):
    """轨迹按时间插值，返回 (x, y)"""
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
    x0, y0 = interp_traj(traj, t - 3.0)
    x1, y1 = interp_traj(traj, t + 3.0)
    if abs(x1 - x0) + abs(y1 - y0) < 0.05:
        return None
    return float(np.arctan2(y1 - y0, x1 - x0))


def field_at(arr, t, T):
    """三维时序场在时刻 t 的双帧线性插值"""
    k = int(np.searchsorted(T, t))
    if k <= 0:
        return arr[0]
    if k >= len(T):
        return arr[-1]
    a = (t - T[k - 1]) / max(T[k] - T[k - 1], 1e-9)
    return (1 - a) * arr[k - 1] + a * arr[k]


def load_font(size):
    for f in ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
              "C:/Windows/Fonts/segoeui.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- 点云
def build_static(z, lu, asset):
    parts = [terrain_gaussians(z, lu)]
    for b in asset["static_colliders"]:
        if b["type"] != "building":
            continue
        bx0, by0, bx1, by1 = b["bbox"]
        L, W = bx1 - bx0, by1 - by0
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        parts.append(box_gaussians(cx, cy, b["z_base"], L, W, b["height"],
                                   0.0, (0.92, 0.90, 0.86), dens=1.3))
        parts.append(box_gaussians(cx, cy, b["z_base"] + b["height"] - 0.05,
                                   L + 0.3, W + 0.3, 0.22, 0.0,
                                   PALETTE["roof"], dens=1.0))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def build_dynamic(t, z, metrics, h_field, veh_meta):
    xs, cs, ss, os_ = [], [], [], []

    def add(r):
        if len(r[0]) == 0:
            return
        xs.append(r[0]); cs.append(r[1]); ss.append(r[2]); os_.append(r[3])

    # 车辆
    for i, v in enumerate(metrics["vehicles"]):
        x, y = interp_traj(v["traj"], t)
        yaw = heading_at(v["traj"], t)
        if yaw is None:
            yaw = np.deg2rad(15 * i)
        p = C.VEHICLE_TYPES[v["type"]]
        zb = float(z[int(np.clip(y / C.DY, 0, C.NY - 1)), int(np.clip(x / C.DX, 0, C.NX - 1))])
        base = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
        released = v["released"] and (v["t_release"] is not None) and t >= v["t_release"]
        col = (0.88, 0.22, 0.18) if released else base
        body = box_gaussians(x, y, zb, p["L"], p["W"], p["H"] * 0.55, yaw, col, dens=0.85)
        add((body[0], body[1], body[2] * 0.75, body[3]))
        cab = box_gaussians(x, y, zb + p["H"] * 0.55, p["L"] * 0.55,
                            p["W"] * 0.85, p["H"] * 0.42, yaw, col, dens=0.85)
        add((cab[0], cab[1], cab[2] * 0.75, cab[3]))

    # 货物：完全浸没后浮起，浮起后略微升高
    for b in metrics["cargo_boxes"]:
        x, y = interp_traj(b["traj"], t)
        zb = float(z[int(np.clip(y / C.DY, 0, C.NY - 1)), int(np.clip(x / C.DX, 0, C.NX - 1))])
        lift = 0.35 if (b["released"] and b["t_release"] is not None and t >= b["t_release"]) else 0.0
        add(box_gaussians(x, y, zb + lift, 0.8, 0.8, 0.8, 0.0,
                          PALETTE["cargo"], dens=0.55))

    # 人员：失稳后变橙
    for hm in metrics["humans"]:
        x, y = hm["x"], hm["y"]
        zb = float(z[int(np.clip(y / C.DY, 0, C.NY - 1)), int(np.clip(x / C.DX, 0, C.NX - 1))])
        col = PALETTE["human"] if hm["level"] == 0 else (1.0, 0.55, 0.0)
        add(box_gaussians(x, y, zb, 0.45, 0.28, 1.72, 0.0, col, dens=0.42))

    # 水面
    add(water_gaussians(h_field, z, veh_meta["lu"]))

    return tuple(np.concatenate(lst) for lst in (xs, cs, ss, os_))


# ---------------------------------------------------------------- 叠加层
def overlay(img, t, t_end, hmax, n_rel, total_rel, font, small):
    """在渲染帧上叠加时间戳、水位与进度条"""
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size

    # 左上信息板
    dr.rounded_rectangle([14, 12, 342, 92], radius=8, fill=(10, 32, 52, 208))
    dr.text((26, 20), f"t = {t:4.0f} s", font=font, fill=(255, 255, 255))
    dr.text((26, 60), f"max depth {hmax:4.2f} m", font=small, fill=(186, 218, 242))
    dr.text((196, 60), f"failed {n_rel}/{total_rel}", font=small,
            fill=(255, 168, 116) if n_rel else (186, 218, 242))

    # 底部进度条
    y0 = H - 26
    dr.rectangle([0, y0 - 4, W, H], fill=(10, 32, 52, 190))
    dr.rectangle([0, y0 - 4, int(W * min(t / t_end, 1.0)), y0 - 1], fill=(240, 160, 80, 255))
    for frac, lab in [(0.0, "0"), (0.5, f"{t_end/2:.0f}"), (1.0, f"{t_end:.0f} s")]:
        dr.text((min(max(W * frac - 8, 6), W - 40), y0 + 2), lab, font=small,
                fill=(200, 210, 220))
    return np.asarray(im)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=5.0, help="帧间隔秒数")
    ap.add_argument("--tend", type=float, default=600.0, help="动画结束时刻")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--scale", type=float, default=1.0, help="分辨率缩放")
    ap.add_argument("--cam", default="wide", choices=["wide", "close"],
                    help="wide=全景；close=停车场近景（看车辆被冲走）")
    ap.add_argument("--out", default="outputs/flood_impact.mp4")
    args = ap.parse_args()

    d = np.load(C.F_SCENE)
    z = d["z"].astype(np.float64); lu = d["landuse"]
    run = np.load(C.F_RUN)
    T = run["t"].astype(np.float64)
    Hseq = run["h"]
    metrics = json.load(open(C.F_METRICS, encoding="utf-8"))
    asset = json.load(open(C.ASSETS / "scene_asset.json", encoding="utf-8"))

    # 两个机位：全景看整体淹没过程；近景看单个车辆的失稳与漂移
    CAMS = {
        "wide":  (C.CAMERA["eye"], C.CAMERA["target"]),
        "close": ((116.0, 58.0, 118.0), (112.0, 96.0, 90.0)),
    }
    cam_eye, cam_tgt = CAMS[args.cam]

    W = int(C.IMG_W * args.scale); Hh = int(C.IMG_H * args.scale)
    font = load_font(int(26 * args.scale)); small = load_font(int(15 * args.scale))

    print("[anim] 构建静态场景点云 ...")
    st = build_static(z, lu, asset)
    print(f"[anim] 静态 {len(st[0])} 个高斯")

    times = np.arange(0, args.tend + 1e-6, args.dt)
    total = sum(1 for v in metrics["vehicles"] if v["released"])
    frames, t_start = [], time.time()

    for n, t in enumerate(times):
        h_field = field_at(Hseq, t, T).astype(np.float64)
        dyn = build_dynamic(t, z, metrics, h_field,
                            dict(lu=lu))
        xyz = np.concatenate([st[0], dyn[0]])
        rgb = np.concatenate([st[1], dyn[1]])
        scl = np.concatenate([st[2], dyn[2]])
        opa = np.concatenate([st[3], dyn[3]])

        img = project_and_render(xyz, rgb, scl, opa, W, Hh,
                                 cam_eye, cam_tgt, C.FOV_DEG)
        n_rel = sum(1 for v in metrics["vehicles"]
                    if v["released"] and v["t_release"] is not None and t >= v["t_release"])
        img = overlay(img, t, args.tend, float(h_field.max()), n_rel, total, font, small)
        frames.append(img)
        if n % 10 == 0 or n == len(times) - 1:
            el = time.time() - t_start
            eta = el / (n + 1) * (len(times) - n - 1)
            print(f"  帧 {n+1}/{len(times)}  t={t:5.0f}s  "
                  f"高斯 {len(xyz)//1000}k  已用 {el:5.0f}s  剩余 ~{eta:4.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(out, frames, fps=args.fps, quality=8,
                        macro_block_size=None, codec="libx264")
        print(f"[anim] MP4 输出: {out}  ({out.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"[anim] MP4 失败 ({e})，改存 GIF")
        ims = [Image.fromarray(f) for f in frames[::2]]
        gif = out.with_suffix(".gif")
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000/args.fps*2),
                    loop=0, optimize=True)
        print(f"[anim] GIF 输出: {gif}  ({gif.stat().st_size/1e6:.1f} MB)")

    # 同时存一份静帧供 HTML 播放器使用（抽稀 + JPEG 压缩）
    seq_dir = C.OUT / "frames"
    seq_dir.mkdir(exist_ok=True)
    for i, f in enumerate(frames[::2]):
        Image.fromarray(f).resize((W // 1, Hh // 1)).save(
            seq_dir / f"f{i:04d}.jpg", quality=78, optimize=True)
    print(f"[anim] 序列帧: {seq_dir} ({len(list(seq_dir.glob('*.jpg')))} 张)")


if __name__ == "__main__":
    main()
