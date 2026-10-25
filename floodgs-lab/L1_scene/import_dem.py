"""
真实地形导入 —— 把 DEM 转换成 floodgs-lab 的场景资产
=====================================================
支持格式：
    .tif / .tiff   GeoTIFF（优先用 rasterio，没有则用 PIL 读单波段）
    .asc           ESRI ASCII Grid
    .npy           纯 numpy 数组（最省事）

用法：
    python L1_scene/import_dem.py --dem your_dem.tif --dx 1.0
    python L1_scene/import_dem.py --dem your_dem.tif --dx 1.0 \
        --river-mask river.shp  --out assets/scene_real.npz

关键点（踩过的坑）：
  1. GeoTIFF 默认是 north-up，第 0 行是最北。本仓库 y 轴向北为正，
     所以必须垂直翻转，否则整个地形上下颠倒，流向全错。
  2. 重采样用双线性（order=1），不要用最近邻 —— 最近邻会产生台阶，
     水深场会在台阶处出现虚假梯度。
  3. NoData 必须先填成邻域中值，否则会被当成洼地，水全灌进去。
  4. landuse 至少要保证「河道」和「非河道」标记正确，其余标签可以先用默认值。
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

from scipy.ndimage import zoom, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ---------------------------------------------------------------- 读取
def read_dem(path: Path):
    """返回 (z, transform_info)  z 为 float32，南北向已翻转为 y 北为正"""
    suf = path.suffix.lower()
    info = {"source": str(path), "format": suf}

    if suf == ".npy":
        z = np.load(path).astype(np.float64)
        flipped = False

    elif suf in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                z = src.read(1).astype(np.float64)
                info["crs"] = str(src.crs)
                info["transform"] = list(src.transform)[:6]
                if src.nodata is not None:
                    z[z == src.nodata] = np.nan
            flipped = True
        except ImportError:
            from PIL import Image
            im = Image.open(path)
            z = np.array(im, dtype=np.float64)
            flipped = True
            info["note"] = "rasterio 未安装，用 PIL 读取（会丢失地理参考）"

    elif suf == ".asc":
        with open(path, "r") as f:
            header = {}
            for _ in range(6):
                k, v = f.readline().split()
                header[k.lower()] = float(v)
            z = np.loadtxt(f)
        z[z == header.get("nodata_value", -9999)] = np.nan
        info["cellsize"] = header.get("cellsize")
        flipped = True

    else:
        raise ValueError(f"不支持的格式: {suf}（用 .tif/.asc/.npy）")

    if flipped:
        z = z[::-1, :]        # north-up -> y 北为正
        info["flipped"] = True

    # NoData 填充
    n_bad = int(np.isnan(z).sum())
    if n_bad:
        info["nodata_filled"] = n_bad
        med = median_filter(np.nan_to_num(z, nan=np.nanmedian(z)), size=7)
        z = np.where(np.isnan(z), med, z)

    return z.astype(np.float64), info


def resample(z, nx, ny):
    """双线性重采样到目标网格"""
    fy = ny / z.shape[0]
    fx = nx / z.shape[1]
    if abs(fy - 1) < 1e-9 and abs(fx - 1) < 1e-9:
        return z
    return zoom(z, (fy, fx), order=1)


# ---------------------------------------------------------------- 语义推断
def infer_landuse(z, river_mask=None):
    """粗略推断 landuse。
    真正的语义应该来自 3DGS 分割结果；这里只是让 DEM 能跑起来的兜底。
    """
    lu = np.full(z.shape, 2, dtype=np.uint8)      # 默认厂区/平地

    gy, gx = np.gradient(z)
    slope = np.hypot(gx, gy)
    zmin = np.nanmin(z)

    # 高程最低的 30% + 坡度小 -> 河道
    thr = np.nanpercentile(z, 30)
    lu[(z < thr) & (slope < 0.02)] = 0

    # 陡坡 -> 山坡
    lu[slope > 0.15] = 3

    # 用户提供的河道掩膜优先
    if river_mask is not None:
        lu[river_mask] = 0
    return lu


def apply_river_mask(lu, mask_source, z):
    """支持 .npy 布尔掩膜，或按高程阈值生成"""
    if mask_source is None:
        return lu
    p = Path(mask_source)
    if p.exists() and p.suffix == ".npy":
        m = np.load(p).astype(bool)
        if m.shape != lu.shape:
            m = resample(m.astype(float), lu.shape[1], lu.shape[0]) > 0.5
        lu[m] = 0
        print(f"[dem] 河道掩膜: {m.sum()} 个单元")
    else:
        thr = float(mask_source)
        m = z < (np.nanmin(z) + thr)
        lu[m] = 0
        print(f"[dem] 河道掩膜: z < zmin+{thr:.2f} m, {m.sum()} 个单元")
    return lu


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", required=True, help="DEM 文件 (.tif/.asc/.npy)")
    ap.add_argument("--dx", type=float, default=1.0, help="目标网格分辨率 (m)")
    ap.add_argument("--river-mask", default=None,
                    help="河道掩膜：.npy 布尔数组，或一个高程差阈值（米）")
    ap.add_argument("--invert", action="store_true", help="如果地形上下颠倒，加这个参数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.dem)
    if not src.exists():
        print(f"!! 找不到文件: {src}")
        return 1

    print(f"[dem] 读取 {src.name} ...")
    z, info = read_dem(src)
    print(f"[dem] 原始尺寸 {z.shape}, 高程 {np.nanmin(z):.1f} ~ {np.nanmax(z):.1f} m")

    # 按 dx 决定目标网格，保持地形长宽比
    ext_y = z.shape[0] * (info.get("cellsize") or args.dx)
    ext_x = z.shape[1] * (info.get("cellsize") or args.dx)
    ny = max(20, int(round(ext_y / args.dx)))
    nx = max(20, int(round(ext_x / args.dx)))
    # 控制规模：200x200 在本机约 80 s 跑完 900 s 模拟
    if nx * ny > 250_000:
        scale = np.sqrt(250_000 / (nx * ny))
        nx, ny = int(nx * scale), int(ny * scale)
        print(f"[dem] 网格过大，自动降到 {nx}x{ny}（约 {ext_x/ nx:.2f} m 分辨率）")

    z = resample(z, nx, ny)
    if args.invert:
        z = z[::-1, :]
    print(f"[dem] 重采样到 {nx}x{ny}, 分辨率 {ext_x/nx:.2f} m")

    lu = infer_landuse(z)
    lu = apply_river_mask(lu, args.river_mask, z)

    # 坐标系：左上角为原点，x 向东，y 向北
    xc = (np.arange(nx) + 0.5) * (ext_x / nx)
    yc = (np.arange(ny) + 0.5) * (ext_y / ny)

    out = Path(args.out) if args.out else C.ASSETS / "scene_real.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        z=z.astype(np.float32), landuse=lu,
        xc=xc.astype(np.float32), yc=yc.astype(np.float32),
        vehicles=np.zeros((0, 4), np.float32),
        cargo=np.zeros((0, 4), np.float32),
        humans=np.zeros((0, 3), np.float32),
    )
    print(f"[dem] 已保存 {out}")
    print(f"[dem] 类别统计: 河道 {(lu==0).sum()} | 堤防 {(lu==1).sum()} | "
          f"平地 {(lu==2).sum()} | 山坡 {(lu==3).sum()}")

    # 提示后续步骤
    print("\n下一步：")
    print("  1) 在 assets/ 下补一个 scene_asset.json（建筑 bbox、车辆清单）")
    print("  2) 修改 config.py 的 NX/NY/DX/DY 与本文件一致")
    print("  3) 把 config.py 的 F_SCENE 指向本文件，然后跑 run_all.py --fast")


if __name__ == "__main__":
    sys.exit(main() or 0)
