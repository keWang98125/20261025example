"""
3DGS -> 高程场（论文的技术核心之一）
====================================
把 3DGS 重建出来的高斯集合，转换成物理仿真能用的 DEM + 语义掩码 + 物体清单。

为什么不能直接用：3DGS 没有"表面"，只有一堆各向异性的高斯。
物理求解器要的是 z(x,y) 这样的单值高程场，所以必须做投影 + 分位数估计。

核心算法（三步，全部闭式解，不需要神经网络）：
    ① 分桶     每个高斯按 (x,y) 落入水平网格
    ② 分位数   地面 = 低分位（避开树冠/车辆/电线），屋顶 = 高分位
    ③ 补洞     空桶用邻域中值填充

用法：
    python L1_scene/gs2dem.py --ply scene.ply --dx 0.5 --out assets/scene_real.npz
    python L1_scene/gs2dem.py --ply scene.ply --dx 0.5 --scale 1.0 --up z \
        --ground-pct 8 --top-pct 97

注意：
  - 3DGS 的坐标系来自 COLMAP，通常**没有真实尺度**。必须用已知距离标定
    （例如量一段已知长度的墙），通过 --scale 传入。
  - 若重建时相机不是水平放置，z 轴可能与重力方向有夹角。先用 CloudCompare
    或 Meshlab 把点云摆正，再来跑这个脚本（或后续实现 PCA 找地平面）。
"""

import argparse
import json
import struct
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ---------------------------------------------------------------- PLY 解析
_PLY_TYPES = {
    "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
    "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
    "int": "i", "uint": "I", "int32": "i", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def read_ply(path, wanted=("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")):
    """读取 3DGS 输出的 ply，只提取需要的属性，避免几百万点的内存浪费"""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError("不是 ply 文件")
        fmt = None
        n_vert = 0
        props = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("ply 头部不完整")
            t = line.strip().decode("ascii", "ignore")
            if t.startswith("format"):
                fmt = t.split()[1]
            elif t.startswith("element vertex"):
                n_vert = int(t.split()[2])
            elif t.startswith("property") and len(props) >= 0 and n_vert > 0:
                parts = t.split()
                if len(parts) >= 3 and parts[1] != "list":
                    props.append((parts[2], parts[1]))
            elif t == "end_header":
                break

    names = [p[0] for p in props]
    idx = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in idx]
    if missing:
        print(f"[gs2dem] 缺少属性 {missing}，可用的前 15 个: {names[:15]}")
        wanted = tuple(w for w in wanted if w in idx)

    out = {k: np.empty(n_vert, np.float32) for k in wanted}

    if fmt == "ascii":
        with open(path, "r", errors="ignore") as f:
            for _ in range(len(props) + 10):
                if f.readline().strip() == "end_header":
                    break
            for i in range(n_vert):
                v = f.readline().split()
                for k in wanted:
                    out[k][i] = float(v[idx[k]])
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(n, endian + _PLY_TYPES[t_]) for n, t_ in props])
        with open(path, "rb") as f:
            # 重新定位到 end_header 之后
            f.seek(0)
            while f.readline().strip() != b"end_header":
                pass
            data = np.frombuffer(f.read(n_vert * dtype.itemsize), dtype=dtype, count=n_vert)
        for k in wanted:
            out[k] = data[k].astype(np.float32)

    return out


# ---------------------------------------------------------------- 投影
def project_to_grid(xyz, weights, dx, ground_pct=10, top_pct=95,
                    min_weight=0.05, verbose=True):
    """高斯 -> 高程场
    返回 z_ground, z_top, count, extent
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    keep = weights >= min_weight
    x, y, z = x[keep], y[keep], z[keep]
    if verbose:
        print(f"[gs2dem] 有效高斯 {len(z)}/{len(xyz)}（透明度 >= {min_weight}）")

    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    nx = max(10, int(np.ceil((x1 - x0) / dx)))
    ny = max(10, int(np.ceil((y1 - y0) / dy_safe(dx))))

    ix = np.clip(((x - x0) / dx).astype(np.int32), 0, nx - 1)
    iy = np.clip(((y - y0) / dx).astype(np.int32), 0, ny - 1)
    flat = iy.astype(np.int64) * nx + ix

    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    z_s = z[order]

    bounds = np.searchsorted(flat_s, np.arange(nx * ny + 1))
    z_ground = np.full(nx * ny, np.nan, dtype=np.float64)
    z_top = np.full(nx * ny, np.nan, dtype=np.float64)
    counts = np.zeros(nx * ny, dtype=np.int32)

    for cell in range(nx * ny):
        a, b = bounds[cell], bounds[cell + 1]
        if b <= a:
            continue
        vals = z_s[a:b]
        counts[cell] = b - a
        z_ground[cell] = np.percentile(vals, ground_pct)
        z_top[cell] = np.percentile(vals, top_pct)

    z_ground = z_ground.reshape(ny, nx)
    z_top = z_top.reshape(ny, nx)
    counts = counts.reshape(ny, nx)

    if verbose:
        empty = int(np.isnan(z_ground).sum())
        print(f"[gs2dem] 网格 {nx}x{ny}, 空桶 {empty} ({empty/(nx*ny)*100:.1f}%)")

    # 补洞
    if np.isnan(z_ground).any():
        filled = median_filter(np.nan_to_num(z_ground, nan=np.nanmedian(z_ground)), size=5)
        z_ground = np.where(np.isnan(z_ground), filled, z_ground)

    return z_ground, z_top, counts, (x0, y0, x1, y1)


def dy_safe(dx):
    return dx


# ---------------------------------------------------------------- 语义与物体
def split_objects(z_ground, z_top, counts, height_thresh=1.2, min_cells=12):
    """用『高于地面多少』粗略区分地面/建筑/物体。
    真实情况必须用 Gaussian Grouping 或 SAM2 的分割结果，这里只做兜底。
    """
    from scipy.ndimage import binary_dilation
    dh = z_top - z_ground

    # 【关键】建筑 footprint 内没有任何地面高斯，低分位数会被墙面污染，
    # 使"地面高程"虚高数米，进而让 dh 偏小、建筑被漏判。
    # 处理：把建筑区的 z_ground 替换为周边地面的中值。
    mask_bld = dh > height_thresh
    if mask_bld.any():
        ring = binary_dilation(mask_bld, iterations=3) & ~mask_bld
        if ring.sum() > 10:
            fill_val = float(np.median(z_ground[ring]))
            # 用周围地面的相对起伏做一点修正，避免整片完全水平
            z_ground = np.where(mask_bld, fill_val, z_ground)
            dh = z_top - z_ground
            print(f"[gs2dem] 建筑区地面修复: {mask_bld.sum()} 格 -> {fill_val:.2f} m")
    landuse = np.full(z_ground.shape, 2, dtype=np.uint8)     # 默认平地
    landuse[dh > height_thresh] = 5                          # 建筑/高物体
    landuse[counts == 0] = 3                                 # 无数据 -> 当山坡
    gy, gx = np.gradient(z_ground)
    slope = np.hypot(gx, gy)
    landuse[(slope > 0.25) & (dh <= height_thresh)] = 3
    return landuse, dh, z_ground


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--dx", type=float, default=0.5, help="网格分辨率 (m)")
    ap.add_argument("--scale", type=float, default=1.0, help="COLMAP 单位 -> 米 的缩放")
    ap.add_argument("--ground-pct", type=float, default=10.0)
    ap.add_argument("--top-pct", type=float, default=95.0)
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--height-thresh", type=float, default=1.2,
                    help="高于地面多少米算建筑/物体")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flip-y", action="store_true", help="如果南北颠倒")
    args = ap.parse_args()

    src = Path(args.ply)
    if not src.exists():
        print(f"!! 找不到 {src}")
        return 1

    print(f"[gs2dem] 读取 {src.name} ...")
    g = read_ply(src)
    xyz = np.stack([g["x"], g["y"], g["z"]], axis=1).astype(np.float64)
    if args.flip_y:
        xyz[:, 1] *= -1

    # 尺度与基准面
    xyz *= args.scale
    xyz[:, 2] -= np.percentile(xyz[:, 2], 2)      # 最低点归零
    print(f"[gs2dem] 范围 x {xyz[:,0].min():.1f}~{xyz[:,0].max():.1f}, "
          f"y {xyz[:,1].min():.1f}~{xyz[:,1].max():.1f}, "
          f"z {xyz[:,2].min():.1f}~{xyz[:,2].max():.1f} m")

    w = g.get("opacity", np.ones(len(xyz), np.float32))
    w = 1.0 / (1.0 + np.exp(-w.astype(np.float64)))   # 3DGS 的 opacity 是 logit

    z_ground, z_top, counts, ext = project_to_grid(
        xyz, w, args.dx, args.ground_pct, args.top_pct, args.min_opacity)

    landuse, dh, z_ground = split_objects(z_ground, z_top, counts, args.height_thresh)

    out = Path(args.out) if args.out else C.ASSETS / "scene_from_gs.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = z_ground.shape
    xc = ext[0] + (np.arange(nx) + 0.5) * args.dx
    yc = ext[1] + (np.arange(ny) + 0.5) * args.dx

    np.savez_compressed(
        out,
        z=z_ground.astype(np.float32),
        z_top=z_top.astype(np.float32),
        landuse=landuse,
        density=counts.astype(np.int32),
        xc=xc.astype(np.float32), yc=yc.astype(np.float32),
        vehicles=np.zeros((0, 4), np.float32),
        cargo=np.zeros((0, 4), np.float32),
        humans=np.zeros((0, 3), np.float32),
    )

    # 精度自检报告
    dense = counts > 5
    print(f"\n[gs2dem] 输出 {out}")
    print(f"[gs2dem] 网格 {nx}x{ny} @ {args.dx} m, 有效格 {dense.sum()} "
          f"({dense.mean()*100:.1f}%)")
    print(f"[gs2dem] 地面高程 {np.nanmin(z_ground):.2f} ~ {np.nanmax(z_ground):.2f} m")
    print(f"[gs2dem] 高于地面 >{args.height_thresh} m 的面积 "
          f"{(dh > args.height_thresh).sum() * args.dx**2:.0f} m²")

    print("""
[gs2dem] 必做的三项校核（写进论文的方法章节）：
  1) 高程精度：找 5-10 个已知高程点（井盖、台阶、水准点），
     计算 RMSE。目标 < 0.10 m。这一步决定水深精度。
  2) 密度检查：density 低的区域（< 5 个高斯）说明重建有空洞，
     这些地方的水深不可信，要在论文里标注。
  3) 语义校核：landuse 的准确性直接决定绕流形态，
     随机抽 50 个格子人工核对，报混淆矩阵。
""")


if __name__ == "__main__":
    sys.exit(main() or 0)
