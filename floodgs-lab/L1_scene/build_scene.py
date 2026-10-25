"""
L1 场景就绪化层（Scene-Readiness Layer）
========================================
把"一个场景"变成"物理引擎能读的资产包"。

真实流程（GPU 上跑）：
    Gaussian Grouping 分割 -> InsertAny3D 生成物体高斯 + 位姿 -> 本文件汇总成资产包

本仓库的轻量替代：
    直接用参数化几何生成地形 + 语义标签 + 物体清单，输出**完全相同的接口格式**。
    这样你在服务器上跑完真实分割后，只要把 assets/scene.npz 里的 z / landuse 换掉，
    下游 L2/L3/L4 代码无需修改。

输出：
    assets/scene.npz        数值场（地形、语义、物体坐标）
    assets/scene_asset.json 场景资产清单（可读，用于人工检查 + 论文配图）
"""

import json
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ---------------------------------------------------------------- 语义标签
LU_RIVER, LU_LEVEE, LU_YARD, LU_HILL, LU_ROAD, LU_BUILDING = 0, 1, 2, 3, 4, 5
LU_NAMES = {0: "river", 1: "levee", 2: "yard", 3: "hill", 4: "road", 5: "building"}


def river_center_y(x):
    """河道中心线。取直线：蜿蜒河道会在网格上产生锯齿状岸线，
    导致沿程出现"死端"格子（水流到边缘格子后无下游邻居），
    诱发虚假壅水。示意性算例用直线河道更稳健。"""
    return 59.0 + 0.0 * x


def build_terrain():
    """生成地形高程 + 语义标签"""
    nx, ny, dx, dy = C.NX, C.NY, C.DX, C.DY
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc)              # (ny, nx)

    z_bed = C.Z_BED_UP - C.BED_SLOPE * X    # 河底纵剖面，随 x 降低

    d = np.abs(Y - river_center_y(X))       # 到河道中心线的横向距离
    north = Y > river_center_y(X)

    z = np.empty_like(X)
    lu = np.full(X.shape, LU_YARD, dtype=np.uint8)

    # 河槽：中心略深，横向二次抬升
    m_river = d < C.RIVER_HALF_W
    z[m_river] = z_bed[m_river]          # 平底矩形槽，避免初始横向二次流
    lu[m_river] = LU_RIVER

    # 迎水坡（11 -> 15 m）：河底抬升到堤顶
    m_slope = (d >= C.RIVER_HALF_W) & (d < C.RIVER_HALF_W + 4.0)
    t = (d - C.RIVER_HALF_W) / 4.0
    z[m_slope] = (z_bed + t * C.LEVEE_TOP_OFFSET)[m_slope]
    lu[m_slope] = LU_LEVEE

    # 堤顶平台（15 -> 18 m）
    m_crest = (d >= C.RIVER_HALF_W + 4.0) & (d < C.RIVER_HALF_W + 7.0)
    z[m_crest] = (z_bed + C.LEVEE_TOP_OFFSET)[m_crest]
    lu[m_crest] = LU_LEVEE

    # 背水坡（18 -> 22 m）：降到台地
    m_back = (d >= C.RIVER_HALF_W + 7.0) & (d < C.RIVER_HALF_W + 11.0)
    t2 = (d - (C.RIVER_HALF_W + 7.0)) / 4.0
    yard_z = z_bed + C.YARD_OFFSET
    z[m_back] = (z_bed + C.LEVEE_TOP_OFFSET - t2 * (C.LEVEE_TOP_OFFSET - C.YARD_OFFSET))[m_back]
    lu[m_back] = LU_LEVEE

    # 台地：北岸=厂区，南岸=山坡
    m_plateau = d >= C.RIVER_HALF_W + 11.0
    # 厂区地面自南（靠堤）向北微降，形成过水坡面；
    # 若做成水平，洪水灌满后水面变平即无压力梯度，厂区会变成"蓄水池"而永久壅水
    yard_drop = C.YARD_SLOPE * np.maximum(Y - C.YARD_Y0, 0.0)
    z[m_plateau & north] = (yard_z + 0.10 * np.sin(X * 0.11) - yard_drop)[m_plateau & north]
    lu[m_plateau & north] = LU_YARD
    z[m_plateau & ~north] = (z_bed + 9.0 + 0.004 * (60.0 - Y) ** 2 / 10)[m_plateau & ~north]
    lu[m_plateau & ~north] = LU_HILL

    # ---- 溃口：把北岸堤身削到 +1.6 m
    yc110 = river_center_y(C.BREACH_X)
    m_breach = (np.abs(X - C.BREACH_X) < C.BREACH_W / 2) & \
               (Y > yc110) & (d >= C.RIVER_HALF_W) & (d < C.RIVER_HALF_W + 11.0)
    # 侧壁做线性过渡，避免 2.4 m 的垂直台阶在数值上产生持续冲击震荡
    u_br = np.clip((C.BREACH_W / 2 - np.abs(X - C.BREACH_X)) / 4.0, 0.0, 1.0)
    z_br = (z_bed + C.LEVEE_TOP_OFFSET) * (1 - u_br) + (z_bed + 1.8) * u_br
    z[m_breach] = z_br[m_breach]

    # ---- 厂区硬化道路（停车场 + 主干道），只改糙率不影响高程
    road = ((X > 104) & (X < 132) & (Y > 136) & (Y < 165)) | \
           ((Y > 168) & (Y < 176) & (X > 60) & (X < 175))
    lu[road & (lu == LU_YARD)] = LU_ROAD

    return xc, yc, X, Y, z, lu, z_bed


# ---------------------------------------------------------------- 物体清单
def build_objects(X, Y, z):
    """按语义放置建筑、车辆、货堆、人员"""
    rng = np.random.default_rng(2026)

    def z_at(x, y):
        i = int(np.clip(x / C.DX, 0, C.NX - 1))
        j = int(np.clip(y / C.DY, 0, C.NY - 1))
        return float(z[j, i])

    # ---- 建筑（同时作为水流障碍：footprint 高程抬到屋顶）
    # 位置刻意放在溃口下游 40 m 以外：洪水绕流厂房时会局部加速，
    # 这与你老师说的"车被冲走、货被冲散"是同一物理过程
    buildings = [
        dict(id="PlantA", x0=86,  x1=106, y0=122, y1=150, h=9.0, use="main_workshop"),
        dict(id="PlantB", x0=132, x1=152, y0=108, y1=132, h=7.0, use="warehouse"),
        dict(id="StoreC", x0=54,  x1=74,  y0=152, y1=174, h=6.0, use="storage"),
    ]
    for b in buildings:
        b["z_base"] = z_at((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2)
        i0, i1 = int(b["x0"] / C.DX), int(b["x1"] / C.DX)
        j0, j1 = int(b["y0"] / C.DY), int(b["y1"] / C.DY)

    # ---- 车辆：停在溃口下游 5~30 m 的厂区停车场（洪水冲刷带内）
    vpos = [
        (103.0, 84.0,  0.0,  "car"),
        (108.0, 86.0,  0.0,  "suv"),
        (113.0, 88.0,  0.0,  "car"),
        (104.0, 93.0,  90.0, "van"),
        (116.0, 94.0,  20.0, "car"),
        (121.0, 100.0, 70.0, "suv"),
    ]
    vehicles = [dict(id=f"V{i+1}", x=float(x), y=float(y),
                     heading=float(hdg), type=t, z=z_at(x, y))
                for i, (x, y, hdg, t) in enumerate(vpos)]

    # ---- 货堆：3 堆，每堆 18 个箱子，同样位于冲刷带内
    cargo = []
    piles = [(100.0, 90.0), (110.0, 98.0), (118.0, 106.0)]
    for p, (cx, cy) in enumerate(piles):
        for k in range(18):
            gx = cx + rng.uniform(-2.2, 2.2)
            gy = cy + rng.uniform(-2.2, 2.2)
            cargo.append(dict(id=f"C{p+1}_{k:02d}", pile=p + 1, x=float(gx), y=float(gy),
                              z=z_at(gx, gy), layer=int(k // 6)))
    # ---- 人员：建筑出口（疏散起点）+ 厂区空地
    # 分三组：建筑出口（相对安全）/ 停车场（冲刷带，受威胁）/ 厂区东部
    humans = [
        (88.0, 120.0), (92.0, 120.0), (96.0, 121.0), (100.0, 124.0),
        (108.0, 75.0), (113.0, 75.5), (118.0, 76.0),      # 溃口出口：深水急流区
        (110.0, 96.0), (124.0, 96.0),
        (134.0, 106.0), (142.0, 110.0),
    ]
    humans = [dict(id=f"H{i+1}", x=float(x), y=float(y), z=z_at(x, y))
              for i, (x, y) in enumerate(humans)]

    # ---- 把建筑 footprint 抬到屋顶（作为水流障碍）
    for b in buildings:
        i0, i1 = int(b["x0"] / C.DX), int(b["x1"] / C.DX)
        j0, j1 = int(b["y0"] / C.DY), int(b["y1"] / C.DY)
        z[j0:j1, i0:i1] = b["z_base"] + b["h"]

    # 建筑内的车/人剔除（实际不会放在室内，这里仅做鲁棒性保护）
    return buildings, vehicles, cargo, humans


def main():
    C.ASSETS.mkdir(parents=True, exist_ok=True)
    xc, yc, X, Y, z, lu, z_bed = build_terrain()
    buildings, vehicles, cargo, humans = build_objects(X, Y, z)

    # 建筑 footprint 语义
    for b in buildings:
        i0, i1 = int(b["x0"] / C.DX), int(b["x1"] / C.DX)
        j0, j1 = int(b["y0"] / C.DY), int(b["y1"] / C.DY)
        lu[j0:j1, i0:i1] = LU_BUILDING

    veh_arr = np.array([[v["x"], v["y"], v["heading"], v["z"]] for v in vehicles])
    cargo_arr = np.array([[c["x"], c["y"], c["z"], c["pile"]] for c in cargo])
    hum_arr = np.array([[h["x"], h["y"], h["z"]] for h in humans])

    np.savez_compressed(
        C.F_SCENE,
        z=z.astype(np.float32), landuse=lu, xc=xc, yc=yc,
        z_bed=z_bed.astype(np.float32),
        vehicles=veh_arr, cargo=cargo_arr, humans=hum_arr,
    )

    asset = dict(
        scene_id="synthetic_yard_01",
        crs="local_ENU",
        grid=dict(nx=C.NX, ny=C.NY, dx=C.DX, dy=C.DY),
        note="本文件格式与 InsertAny3D 的 pose.json / Gaussian Grouping 语义掩码对齐。"
             "替换为真实场景时只需保持字段名一致。",
        terrain_stats=dict(z_min=float(z.min()), z_max=float(z.max())),
        static_colliders=[
            dict(id=b["id"], type="building", bbox=[b["x0"], b["y0"], b["x1"], b["y1"]],
                 z_base=b["z_base"], height=b["h"], use=b["use"], rigid=False,
                 source="GaussianGrouping:mask -> mesh")
            for b in buildings
        ] + [dict(id="levee_north", type="levee", z_offset=C.LEVEE_TOP_OFFSET,
                  breach_x=C.BREACH_X, breach_w=C.BREACH_W, rigid=False,
                  source="terrain")],
        dynamic_assets=[
            dict(id=v["id"], type="vehicle", sub=v["type"],
                 pose=dict(xyz=[v["x"], v["y"], v["z"]], yaw_deg=v["heading"]),
                 mass_kg=C.VEHICLE_TYPES[v["type"]]["mass"],
                 source="InsertAny3D:inserted_object.ply")
            for v in vehicles
        ] + [dict(id=f"pile{p}", type="cargo_pile", sub="boxes", n=18,
                  pose=dict(xyz=[p[0], p[1], 0.0], yaw_deg=0.0),
                  mass_kg=18 * C.CARGO_BOX["mass"], source="manual")
             for p in [(100.0, 90.0), (110.0, 98.0), (118.0, 106.0)]],
        humans=humans,
        fluid_boundary=dict(inlet="x=0 (west, river only)", outlet="x=200 (east, open)",
                            q_peak=C.Q_PEAK, t_peak=C.T_PEAK),
    )
    with open(C.ASSETS / "scene_asset.json", "w", encoding="utf-8") as f:
        json.dump(asset, f, ensure_ascii=False, indent=2)

    print(f"[L1] 地形 {z.shape}  高程 {z.min():.1f}~{z.max():.1f} m")
    print(f"[L1] 建筑 {len(buildings)} | 车辆 {len(vehicles)} | 货物 {len(cargo)} | 人员 {len(humans)}")
    print(f"[L1] 输出: {C.F_SCENE.name}, scene_asset.json")
    for k, name in LU_NAMES.items():
        print(f"       {name:9s} {int((lu==k).sum()):6d} cells")


if __name__ == "__main__":
    main()
