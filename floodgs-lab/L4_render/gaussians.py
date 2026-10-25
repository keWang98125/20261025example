"""
L4-1 从场景资产生成 3D 高斯点云
================================
这一层在真实流程里由 Gaussian Grouping / InsertAny3D 输出 .ply，
本文件用参数化几何生成等价的高斯集合，字段结构与 3DGS 的 ply 完全一致：
    (x, y, z, f_dc_r, f_dc_g, f_dc_b, opacity, scale)

生成四类图元：
    terrain  —— 地形表面（按 landuse 着色）
    building —— 建筑墙面 + 屋顶
    object   —— 车辆 / 货箱 / 人员
    water    —— 水面（洪水淹没区，半透明）
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# 基础色（线性近似，直接当 sRGB 用即可）
PALETTE = {
    "hill":      (0.28, 0.38, 0.20),
    "yard":      (0.60, 0.58, 0.51),
    "road":      (0.33, 0.33, 0.35),
    "levee":     (0.60, 0.53, 0.40),
    "riverbed":  (0.40, 0.42, 0.40),
    "wall":      (0.80, 0.78, 0.72),
    "roof":      (0.42, 0.27, 0.24),
    "cargo":     (0.62, 0.48, 0.28),
    "human":     (0.85, 0.25, 0.25),
}
VEHICLE_COLORS = [
    (0.80, 0.80, 0.82), (0.15, 0.25, 0.55), (0.72, 0.15, 0.15),
    (0.90, 0.88, 0.84), (0.20, 0.45, 0.30), (0.35, 0.35, 0.38),
]
LIGHT = np.array([0.45, 0.30, 0.84]); LIGHT /= np.linalg.norm(LIGHT)


def _shade(normals):
    return 0.55 + 0.45 * np.clip(normals @ LIGHT, 0, 1)


def _pack(pts, cols, scales, opac, out):
    out["xyz"].append(pts.astype(np.float32))
    out["rgb"].append(cols.astype(np.float32))
    out["scale"].append(np.asarray(scales, dtype=np.float32).reshape(-1))
    out["opacity"].append(np.asarray(opac, dtype=np.float32).reshape(-1))


def terrain_gaussians(z, lu, step=2):
    """地形表面高斯：2 m 间隔采样，排除建筑 footprint"""
    ny, nx = z.shape
    jj, ii = np.meshgrid(np.arange(0, ny, step), np.arange(0, nx, step), indexing="ij")
    keep = lu[jj, ii] != 5
    jj, ii = jj[keep], ii[keep]
    x = (ii + 0.5) * C.DX
    y = (jj + 0.5) * C.DY
    zz = z[jj, ii]
    # 法向
    dzdx = np.gradient(z, axis=1)[jj, ii]
    dzdy = np.gradient(z, axis=0)[jj, ii]
    n = np.stack([-dzdx, -dzdy, np.ones_like(dzdx)], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    lut = {3: "hill", 0: "riverbed", 1: "levee", 2: "yard", 4: "road"}
    cols = np.array([PALETTE[lut.get(int(l), "yard")] for l in lu[jj, ii]], dtype=np.float32)
    # 轻微色差，避免大面积纯色
    rng = np.random.default_rng(7)
    cols *= (0.92 + 0.16 * rng.random((len(cols), 1))).astype(np.float32)
    cols *= _shade(n).astype(np.float32)[:, None]
    pts = np.stack([x, y, zz], axis=1)
    return pts, cols, np.full(len(pts), 1.45, dtype=np.float32), np.full(len(pts), 0.95, dtype=np.float32)


def box_gaussians(x0, y0, z0, L, W, H, yaw, color, dens=1.1, opacity=0.98):
    """一个长方体的 6 个面采样成高斯"""
    pts, cols, scl, opa = [], [], [], []
    c, s = np.cos(yaw), np.sin(yaw)

    def emit(u, v, w, nx, ny, nz):
        """局部坐标 (u沿L, v沿W, w沿H) -> 世界"""
        px = x0 + c * u - s * v
        py = y0 + s * u + c * v
        pz = z0 + w
        nrm = np.stack([c * np.full_like(u, nx) - s * np.full_like(v, ny),
                        s * np.full_like(u, nx) + c * np.full_like(v, ny),
                        np.full_like(w, nz)], axis=1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        sh = _shade(nrm)
        col = np.tile(np.array(color), (len(u), 1)) * sh[:, None]
        pts.append(np.stack([px, py, pz], axis=1))
        cols.append(col.astype(np.float32))
        scl.append(np.full(len(u), dens * 0.62, dtype=np.float32))
        opa.append(np.full(len(u), opacity, dtype=np.float32))

    # 4 个侧面
    nL = max(2, int(L / dens)); nW = max(2, int(W / dens)); nH = max(2, int(H / dens))
    uu = np.linspace(-L / 2, L / 2, nL)
    vv = np.linspace(-W / 2, W / 2, nW)
    ww = np.linspace(0, H, nH)
    U, Wg = np.meshgrid(uu, ww)
    emit(U.ravel(), np.full(U.size, -W / 2), Wg.ravel(), 0, -1, 0)
    emit(U.ravel(), np.full(U.size, W / 2), Wg.ravel(), 0, 1, 0)
    V, Wg2 = np.meshgrid(vv, ww)
    emit(np.full(V.size, -L / 2), V.ravel(), Wg2.ravel(), -1, 0, 0)
    emit(np.full(V.size, L / 2), V.ravel(), Wg2.ravel(), 1, 0, 0)
    # 顶面
    U, V = np.meshgrid(uu, vv)
    emit(U.ravel(), V.ravel(), np.full(U.size, H), 0, 0, 1)
    return (np.concatenate(pts), np.concatenate(cols),
            np.concatenate(scl), np.concatenate(opa))


def water_gaussians(h, z, lu, level=0.08, step=2, t_scale=1.0):
    """淹没水面：扁平半透明高斯，颜色随水深加深"""
    ny, nx = h.shape
    jj, ii = np.meshgrid(np.arange(0, ny, step), np.arange(0, nx, step), indexing="ij")
    wet = (h[jj, ii] > level) & (lu[jj, ii] != 5)
    jj, ii = jj[wet], ii[wet]
    if len(jj) == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.float32))
    x = (ii + 0.5) * C.DX
    y = (jj + 0.5) * C.DY
    zz = z[jj, ii] + h[jj, ii]
    d = h[jj, ii]
    # 浅水偏青绿，深水偏靛蓝
    tt = np.clip(d / 1.5, 0, 1)[:, None]
    c_shallow = np.array([0.32, 0.58, 0.66])
    c_deep = np.array([0.05, 0.18, 0.40])
    cols = (c_shallow * (1 - tt) + c_deep * tt).astype(np.float32)
    opacity = np.clip(0.58 + 0.28 * np.clip(d / 1.2, 0, 1), 0.55, 0.90)
    pts = np.stack([x, y, zz - 0.03], axis=1)
    return pts, cols, np.full(len(pts), 1.45 * step / 2, dtype=np.float32), opacity.astype(np.float32)


def build_scene_cloud(vehicle_states=None, cargo_states=None, human_states=None,
                      water=True, h_field=None, z=None, lu=None):
    """组装完整点云
    vehicle_states: [(x, y, yaw, type_idx), ...] 若为 None 用初始位置
    """
    out = {"xyz": [], "rgb": [], "scale": [], "opacity": []}

    d = np.load(C.F_SCENE)
    if z is None:
        z = d["z"].astype(np.float64)
    if lu is None:
        lu = d["landuse"]

    p, c, s, o = terrain_gaussians(z, lu)
    _pack(p, c, s, o, out)

    # 建筑
    import json
    asset = json.load(open(C.ASSETS / "scene_asset.json", encoding="utf-8"))
    for b in asset["static_colliders"]:
        if b["type"] != "building":
            continue
        bx0, by0, bx1, by1 = b["bbox"]
        L, W = bx1 - bx0, by1 - by0
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        p, c, s, o = box_gaussians(cx, cy, b["z_base"], L, W, b["height"],
                                   0.0, (0.92, 0.90, 0.86), dens=1.3)
        # 墙面用浅色、屋顶用深色：这里统一给墙色，顶面单独再上一层
        _pack(p, c, s, o, out)
        p, c, s, o = box_gaussians(cx, cy, b["z_base"] + b["height"] - 0.05,
                                   L + 0.3, W + 0.3, 0.22, 0.0, PALETTE["roof"], dens=1.0)
        _pack(p, c, s, o, out)

    # 车辆
    if vehicle_states is None:
        vehicle_states = []
        for i, v in enumerate(asset["dynamic_assets"]):
            if v["type"] == "vehicle":
                vehicle_states.append((v["pose"]["xyz"][0], v["pose"]["xyz"][1],
                                       np.deg2rad(v["pose"]["yaw_deg"]), v["sub"], i))
    for k, (vx, vy, yaw, vt, idx) in enumerate(vehicle_states):
        p = C.VEHICLE_TYPES[vt]
        zb = float(z[int(np.clip(vy / C.DY, 0, C.NY - 1)), int(np.clip(vx / C.DX, 0, C.NX - 1))])
        col = VEHICLE_COLORS[idx % len(VEHICLE_COLORS)]
        pp, cc, ss, oo = box_gaussians(vx, vy, zb, p["L"], p["W"], p["H"] * 0.55,
                                       yaw, col, dens=0.85)
        _pack(pp, cc, ss * 0.75, oo, out)
        pp, cc, ss, oo = box_gaussians(vx, vy, zb + p["H"] * 0.55, p["L"] * 0.55,
                                       p["W"] * 0.85, p["H"] * 0.42, yaw, col, dens=0.85)
        _pack(pp, cc, ss * 0.75, oo, out)

    # 货物：按箱子逐个放置（初始状态从 scene.npz 读）
    if cargo_states is None:
        cargo_states = [(float(r[0]), float(r[1]), int(r[3])) for r in d["cargo"]]
        cargo_states = [(x, y, 0) for (x, y, _) in cargo_states]
    for (cx, cy, lay) in cargo_states:
        zb = float(z[int(np.clip(cy / C.DY, 0, C.NY - 1)), int(np.clip(cx / C.DX, 0, C.NX - 1))])
        pp, cc, ss, oo = box_gaussians(cx, cy, zb + lay * 0.82, 0.8, 0.8, 0.8,
                                       0.0, PALETTE["cargo"], dens=0.55)
        _pack(pp, cc, ss * 0.85, oo, out)

    # 人员
    if human_states is None:
        human_states = [(float(r[0]), float(r[1]), 0) for r in d["humans"]]
    for (hx, hy, flag) in human_states:
        zb = float(z[int(np.clip(hy / C.DY, 0, C.NY - 1)), int(np.clip(hx / C.DX, 0, C.NX - 1))])
        col = PALETTE["human"] if flag == 0 else (1.0, 0.55, 0.0)
        pp, cc, ss, oo = box_gaussians(hx, hy, zb, 0.45, 0.28, 1.72, 0.0, col, dens=0.42)
        _pack(pp, cc, ss * 0.7, oo, out)

    # 水面
    if water and h_field is not None:
        p, c, s, o = water_gaussians(h_field, z, lu)
        _pack(p, c, s, o, out)

    return (np.concatenate(out["xyz"]), np.concatenate(out["rgb"]),
            np.concatenate(out["scale"]), np.concatenate(out["opacity"]))
