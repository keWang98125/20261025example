"""
L3 灾害评估层 —— 三套可验证的状态级指标
=========================================
创新点三的落地：把"灾害仿真"从渲染动画变成**灾前/灾后状态标签 + 可核对指标**。

  1) 车辆：浮力-拖曳-摩阻三方平衡判据 -> 失稳分类 -> 拉格朗日输运
     对标：Xiong et al. (2024, WRR) SWE-DEM 全耦合
  2) 货物：箱体群散开范围（凸包面积 / IoU）
  3) 人员：Milanesi et al. (2015, WRR) 三模式失稳判据 -> 风险栅格

判据来源与诚实声明：
  - 车辆判据是**物理平衡的简化实现**（非 Xiong 的 DEM 接触模型）。
    论文中应写"采用基于浮力-拖曳平衡的简化刚体判据，与 Xiong et al. (2024)
    的 DEM 结果对比校准"，而不是声称复现了 SWE-DEM。
  - 人体模型复现了 Milanesi 的核心力矩平衡思路，参数取自其论文正文。
    若要严格对比，需下载其附录代码核对系数（docs/实验方案.md 有说明）。
"""

import json
import numpy as np
from scipy.spatial import ConvexHull

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


# ================================================================ 场采样
class FieldSampler:
    """在快照序列上做时空线性插值"""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.t = d["t"].astype(np.float64)
        self.h, self.ux, self.uy = d["h"], d["ux"], d["uy"]
        self.z = d["z"]
        self.landuse = d["landuse"]
        self.dx, self.dy = C.DX, C.DY

    def _bilinear(self, arr, x, y):
        nx, ny = arr.shape[1], arr.shape[0]
        fi = np.clip(x / self.dx - 0.5, 0, nx - 1.001)
        fj = np.clip(y / self.dy - 0.5, 0, ny - 1.001)
        i0 = fi.astype(int); j0 = fj.astype(int)
        wi = fi - i0; wj = fj - j0
        return (arr[j0, i0] * (1 - wi) * (1 - wj) + arr[j0, i0 + 1] * wi * (1 - wj)
                + arr[j0 + 1, i0] * (1 - wi) * wj + arr[j0 + 1, i0 + 1] * wi * wj)

    def sample(self, t, x, y):
        k = np.searchsorted(self.t, t)
        k = np.clip(k, 1, len(self.t) - 1)
        a = (t - self.t[k - 1]) / max(self.t[k] - self.t[k - 1], 1e-9)
        a = float(np.clip(a, 0.0, 1.0))
        h = (1 - a) * self._bilinear(self.h[k - 1], x, y) + a * self._bilinear(self.h[k], x, y)
        ux = (1 - a) * self._bilinear(self.ux[k - 1], x, y) + a * self._bilinear(self.ux[k], x, y)
        uy = (1 - a) * self._bilinear(self.uy[k - 1], x, y) + a * self._bilinear(self.uy[k], x, y)
        return float(h), float(ux), float(uy)

    def bed(self, x, y):
        return float(self._bilinear(self.z, x, y))

    def is_wall(self, x, y):
        """是否落在建筑 footprint 内（固壁）"""
        ny, nx = self.landuse.shape
        i = int(np.clip(x / self.dx, 0, nx - 1))
        j = int(np.clip(y / self.dy, 0, ny - 1))
        return bool(self.landuse[j, i] == 5)


# ================================================================ 车辆
def vehicle_forces(h, u, v_rear, p):
    """返回 (F_drag, F_buoy, F_normal, F_friction, submerged_area)"""
    L, W, H, m = p["L"], p["W"], p["H"], p["mass"]
    hs = min(max(h, 0.0), H)
    sub = hs / H
    # 有效排水体积：车厢会进水，取 0.6 折减
    V_sub = 0.60 * L * W * hs
    F_b = C.RHO_W * C.G * V_sub
    # 迎流面积：正对流向，取 W x hs
    A = W * hs
    u_rel = u - v_rear
    F_d = 0.5 * C.RHO_W * p["Cd"] * A * abs(u_rel) * u_rel
    N = max(0.0, m * C.G - F_b)
    return F_d, F_b, N, p["mu"] * N, A


def vehicle_state(h, u, v_rear, p):
    """0=稳定 1=滑移 2=漂浮"""
    F_d, F_b, N, F_f, A = vehicle_forces(h, u, v_rear, p)
    if F_b > p["mass"] * C.G:
        return 2, dict(F_d=abs(F_d), F_b=F_b, F_f=F_f)
    if abs(F_d) > F_f:
        return 1, dict(F_d=abs(F_d), F_b=F_b, F_f=F_f)
    return 0, dict(F_d=abs(F_d), F_b=F_b, F_f=F_f)


def simulate_vehicles(sampler, vehicles, sub_dt=0.5):
    """逐车积分：失稳前锚定，失稳后按拖曳-摩阻平衡输运"""
    out = []
    t_end = float(sampler.t[-1])
    for v in vehicles:
        p = C.VEHICLE_TYPES[v["type"]]
        x, y = v["x"], v["y"]
        vx, vy = 0.0, 0.0
        released = False
        t_rel = None
        mode_at_release = 0
        traj = [(0.0, x, y)]
        hmax_seen, umax_seen = 0.0, 0.0
        t = 0.0
        while t < t_end:
            h, ux, uy = sampler.sample(t, x, y)
            u_mag = float(np.hypot(ux, uy))
            hmax_seen = max(hmax_seen, h)
            umax_seen = max(umax_seen, u_mag)
            # 相对来流（沿运动方向投影，简化为一维）
            if u_mag > 1e-6:
                ex, ey = ux / u_mag, uy / u_mag
            else:
                ex, ey = 0.0, 0.0
            v_along = vx * ex + vy * ey
            st, _ = vehicle_state(h, u_mag, v_along, p)

            if not released and st > 0:
                released = True
                t_rel = t
                mode_at_release = st

            if released:
                Fd, Fb, N, Ff, A = vehicle_forces(h, u_mag, v_along, p)
                # 附加质量：等效质量 = 自重 + Ca * 排开水的质量
                hs_m = min(max(h, 0.0), p["H"])
                V_sub_m = 0.60 * p["L"] * p["W"] * hs_m
                m_eff = p["mass"] + C.ADDED_MASS_COEF * C.RHO_W * V_sub_m
                acc = Fd / m_eff
                # 摩阻阻碍运动
                if abs(v_along) > 1e-6:
                    acc -= np.sign(v_along) * Ff / p["mass"]
                vx += acc * ex * sub_dt
                vy += acc * ey * sub_dt
                # 数值阻尼，避免浅水层的虚假加速
                vx *= 0.995; vy *= 0.995
                sp = np.hypot(vx, vy)
                if sp > 6.0:                       # 车辆漂移物理上限
                    vx, vy = vx / sp * 6.0, vy / sp * 6.0
                nx_, ny_ = x + vx * sub_dt, y + vy * sub_dt
                # 撞建筑：用**车头**探测（车有长度，不能只看中心点，
                # 否则中心停在墙外时车头已插进墙里 2 m）
                sp_ = np.hypot(vx, vy)
                if sp_ > 1e-6:
                    px_ = nx_ + vx / sp_ * p["L"] * 0.5
                    py_ = ny_ + vy / sp_ * p["W"] * 0.5 * 0.0 + vy / sp_ * p["L"] * 0.5
                else:
                    px_, py_ = nx_, ny_
                if sampler.is_wall(px_, py_):
                    vx = vy = 0.0
                # 地形搁浅：水深不足 0.15 m 即触底停住
                elif sampler.sample(t, nx_, ny_)[0] < 0.15:
                    vx = vy = 0.0
                else:
                    x, y = nx_, ny_
                # 漂出计算域 = 被冲出场区，锁定在边界并记终点
                if not (1.0 <= x <= C.NX - 1.0 and 1.0 <= y <= C.NY - 1.0):
                    x = float(np.clip(x, 1.0, C.NX - 1.0))
                    y = float(np.clip(y, 1.0, C.NY - 1.0))
                    vx = vy = 0.0
                    if len(traj) == 1 or (t - traj[-1][0]) >= 2.0:
                        traj.append((t, x, y))
                    break
                if len(traj) == 1 or (t - traj[-1][0]) >= 2.0:
                    traj.append((t, x, y))
            t += sub_dt

        disp = float(np.hypot(x - v["x"], y - v["y"]))
        out.append(dict(
            id=v["id"], type=v["type"],
            x0=v["x"], y0=v["y"], x1=x, y1=y,
            displaced_m=disp,
            released=released, t_release=None if t_rel is None else round(t_rel, 1),
            fail_mode=int(mode_at_release),
            fail_label={0: "stable", 1: "sliding", 2: "floating"}[int(mode_at_release)],
            hmax=round(hmax_seen, 3), umax=round(umax_seen, 3),
            traj=[[round(a, 2) for a in p_] for p_ in traj],
        ))
    return out


# ================================================================ 货物
def simulate_cargo(sampler, cargo, sub_dt=0.5):
    boxes = []
    p = C.CARGO_BOX
    t_end = float(sampler.t[-1])
    for c in cargo:
        x, y = c["x"], c["y"]
        vx = vy = 0.0
        released = False
        t_rel = None
        t = 0.0
        traj_c = [(0.0, x, y)]
        while t < t_end:
            h, ux, uy = sampler.sample(t, x, y)
            hs = min(max(h, 0.0), p["H"])
            V_sub = p["L"] * p["W"] * hs * 0.8
            F_b = C.RHO_W * C.G * V_sub
            u_mag = float(np.hypot(ux, uy))
            floating = F_b > p["mass"] * C.G
            if not released and (floating or hs > 0.6 * p["H"]):
                released = True
                t_rel = t
            if released:
                A = p["W"] * hs
                u_rel = max(u_mag - np.hypot(vx, vy), 0.0)
                Fd = 0.5 * C.RHO_W * p["Cd"] * A * u_rel ** 2
                m_eff = p["mass"] + C.ADDED_MASS_COEF * C.RHO_W * V_sub
                acc = Fd / m_eff
                if u_mag > 1e-6:
                    vx += acc * ux / u_mag * sub_dt
                    vy += acc * uy / u_mag * sub_dt
                vx *= 0.99; vy *= 0.99
                sp = np.hypot(vx, vy)
                if sp > 5.0:
                    vx, vy = vx / sp * 5.0, vy / sp * 5.0
                nx_, ny_ = x + vx * sub_dt, y + vy * sub_dt
                sp_c = np.hypot(vx, vy)
                if sp_c > 1e-6:
                    pcx = nx_ + vx / sp_c * 0.4
                    pcy = ny_ + vy / sp_c * 0.4
                else:
                    pcx, pcy = nx_, ny_
                if sampler.is_wall(pcx, pcy):                   # 撞建筑（用箱体前缘）
                    vx = vy = 0.0
                elif sampler.sample(t, nx_, ny_)[0] < 0.12:     # 搁浅
                    vx = vy = 0.0
                else:
                    x, y = nx_, ny_
                if not (1.0 <= x <= C.NX - 1.0 and 1.0 <= y <= C.NY - 1.0):
                    x = float(np.clip(x, 1.0, C.NX - 1.0))
                    y = float(np.clip(y, 1.0, C.NY - 1.0))
                    break
            if len(traj_c) == 1 or (t - traj_c[-1][0]) >= 3.0:
                traj_c.append((t, x, y))
            t += sub_dt
        out_dom = bool(x <= 1.5 or x >= C.NX - 1.5 or y >= C.NY - 1.5)
        boxes.append(dict(id=c["id"], pile=c["pile"], x0=c["x"], y0=c["y"],
                          x1=round(x, 2), y1=round(y, 2), released=released,
                          out_of_domain=out_dom,
                          t_release=None if t_rel is None else round(t_rel, 1),
                          traj=[[round(a, 2) for a in q] for q in traj_c],
                          displaced=round(float(np.hypot(x - c["x"], y - c["y"])), 1)))
    return boxes


def scatter_metrics(boxes):
    """散开范围：灾前/灾后凸包面积比 + 同堆质心位移"""
    res = {}
    by_pile = {}
    for b in boxes:
        by_pile.setdefault(b["pile"], []).append(b)
    for pid, bs in by_pile.items():
        P0 = np.array([[b["x0"], b["y0"]] for b in bs])
        P1 = np.array([[b["x1"], b["y1"]] for b in bs])
        def area(P):
            if len(P) < 3:
                return 0.0
            try:
                return float(ConvexHull(P).volume)   # 2D 时 volume 即面积
            except Exception:
                return 0.0
        a0, a1 = area(P0), area(P1)
        c0, c1 = P0.mean(0), P1.mean(0)
        # RMS 半径：比凸包面积稳健得多（凸包在箱体共线时会退化为 0）
        r0 = float(np.sqrt(((P0 - c0) ** 2).sum(1).mean()))
        r1 = float(np.sqrt(((P1 - c1) ** 2).sum(1).mean()))
        # 归一化散开度：灾后 / 灾前（灾前为 0 时记为 nan）
        res[f"pile{pid}"] = dict(
            n=len(bs),
            n_released=sum(1 for b in bs if b["released"]),
            area_before_m2=round(a0, 1),
            area_after_m2=round(a1, 1),
            rms_radius_before_m=round(r0, 2),
            rms_radius_after_m=round(r1, 2),
            spread_ratio=round(r1 / max(r0, 1e-6), 2),
            centroid_shift_m=round(float(np.hypot(*(c1 - c0))), 1),
            mean_displacement_m=round(float(np.mean([b["displaced"] for b in bs])), 1),
        )
    return res


# ================================================================ 人员
def human_risk_map(sampler, t_end=None, dt=15.0):
    """Milanesi (2015) 三模式判据，逐帧计算取过程最大值
    0=安全 1=滑移 2=倾覆 3=溺水/漂浮
    """
    P = C.HUMAN
    ny, nx = sampler.h.shape[1], sampler.h.shape[2]
    risk = np.zeros((ny, nx), dtype=np.uint8)
    peak_h = np.zeros((ny, nx), dtype=np.float32)
    peak_u = np.zeros((ny, nx), dtype=np.float32)
    ts = sampler.t if t_end is None else sampler.t[sampler.t <= t_end]

    for t in ts:
        k = int(np.searchsorted(sampler.t, t))
        k = min(k, len(sampler.t) - 1)
        h = sampler.h[k].astype(np.float64)
        u = np.hypot(sampler.ux[k], sampler.uy[k]).astype(np.float64)
        peak_h = np.maximum(peak_h, h)
        peak_u = np.maximum(peak_u, u)

        hs = np.minimum(h, P["height"])
        V_sub = 0.45 * P["width"] * P["depth"] * hs          # 人体形状折减
        F_b = C.RHO_W * C.G * V_sub
        W_eff = np.maximum(P["mass"] * C.G - F_b, 0.0)
        F_f = P["mu"] * W_eff
        A_proj = P["width"] * hs
        F_d = 0.5 * C.RHO_W * P["Cd"] * A_proj * u ** 2
        M_d = F_d * (hs * 0.5)
        M_s = W_eff * (P["depth"] * 0.5)

        r = np.zeros_like(risk)
        wet = h > 0.05
        r[wet & (F_d > F_f)] = 1
        r[wet & (M_d > M_s)] = 2
        r[wet & ((h >= 0.85 * P["height"]) | (F_b >= P["mass"] * C.G))] = 3
        risk = np.maximum(risk, r.astype(np.uint8))
    return risk, peak_h, peak_u


def sample_human_points(sampler, humans):
    """对每个具体人员位置做时间包络判定（逐帧算风险，取最高等级）"""
    P = C.HUMAN
    out = []
    for hm in humans:
        x, y = hm["x"], hm["y"]
        best, best_h, best_u = 0, 0.0, 0.0
        for t in sampler.t:
            h, ux, uy = sampler.sample(float(t), x, y)
            um = float(np.hypot(ux, uy))
            hs = min(max(h, 0.0), P["height"])
            V_sub = 0.45 * P["width"] * P["depth"] * hs
            F_b = C.RHO_W * C.G * V_sub
            W_eff = max(P["mass"] * C.G - F_b, 0.0)
            F_f = P["mu"] * W_eff
            A_proj = P["width"] * hs
            F_d = 0.5 * C.RHO_W * P["Cd"] * A_proj * um ** 2
            M_d, M_s = F_d * hs * 0.5, W_eff * P["depth"] * 0.5
            if h < 0.05:
                lvl = 0
            elif h >= 0.85 * P["height"] or F_b >= P["mass"] * C.G:
                lvl = 3
            elif M_d > M_s:
                lvl = 2
            elif F_d > F_f:
                lvl = 1
            else:
                lvl = 0
            if lvl > best:
                best, best_h, best_u = lvl, h, um
            elif lvl == best and h > best_h:
                best_h, best_u = h, um
            # 记录过程极值
            if lvl > 0 and h > best_h:
                best_h = h
        lab = {0: "safe", 1: "sliding", 2: "toppling", 3: "drowning"}[best]
        # 该点的过程最大水深/流速（用于论文表格）
        h_env = 0.0; u_env = 0.0
        for t in sampler.t:
            hh, ux, uy = sampler.sample(float(t), x, y)
            h_env = max(h_env, hh); u_env = max(u_env, float(np.hypot(ux, uy)))
        out.append(dict(id=hm["id"], x=x, y=y, level=int(best), label=lab,
                        h_at_failure=round(best_h, 2), u_at_failure=round(best_u, 2),
                        h_max=round(h_env, 2), u_max=round(u_env, 2)))
    return out


# ================================================================ 主流程
def main():
    d = np.load(C.F_SCENE)
    vehicles = [dict(id=f"V{i+1}", x=float(r[0]), y=float(r[1]), type="car")
                for i, r in enumerate(d["vehicles"])]
    types = ["car", "suv", "car", "van", "car", "suv"]
    for v, t in zip(vehicles, types):
        v["type"] = t

    cargo = [dict(id=f"C{int(r[3])}_{i:02d}", pile=int(r[3]), x=float(r[0]), y=float(r[1]))
             for i, r in enumerate(d["cargo"])]
    humans = [dict(id=f"H{i+1}", x=float(r[0]), y=float(r[1]))
              for i, r in enumerate(d["humans"])]

    print("[L3] 载入流场快照并按时空插值 ...")
    S = FieldSampler(C.F_RUN)
    print(f"[L3] {len(S.t)} 帧, 0~{S.t[-1]:.0f}s")

    print("[L3] 车辆输运仿真 ...")
    veh = simulate_vehicles(S, vehicles)
    print("[L3] 货物输运仿真 ...")
    boxes = simulate_cargo(S, cargo)
    car = scatter_metrics(boxes)
    print("[L3] 人员风险栅格 ...")
    risk, peak_h, peak_u = human_risk_map(S)
    hum_pts = sample_human_points(S, humans)

    n_rel = sum(1 for v in veh if v["released"])
    print(f"[L3] 车辆: {n_rel}/{len(veh)} 失稳, 平均位移 "
          f"{np.mean([v['displaced_m'] for v in veh]):.1f} m")
    for v in veh:
        print(f"      {v['id']}({v['type']:3s}) hmax={v['hmax']:.2f} umax={v['umax']:.2f} "
              f"-> {v['fail_label']:8s} 位移 {v['displaced_m']:5.1f} m")
    print(f"[L3] 货物: {sum(b['released'] for b in boxes)}/{len(boxes)} 起漂")
    for k, v in car.items():
        print(f"      {k}: 面积 {v['area_before_m2']}->{v['area_after_m2']} m2, "
              f"散开比 {v['spread_ratio']}, 质心位移 {v['centroid_shift_m']} m")
    lv = np.array([h["level"] for h in hum_pts])
    print(f"[L3] 人员: 安全 {(lv==0).sum()} | 滑移 {(lv==1).sum()} | "
          f"倾覆 {(lv==2).sum()} | 溺水 {(lv==3).sum()}")

    metrics = dict(
        scene="synthetic_yard_01",
        t_end=float(S.t[-1]),
        vehicles=veh,
        cargo_piles=car,
        cargo_boxes=boxes,
        humans=hum_pts,
        human_risk_stats=dict(
            safe=int((risk == 0).sum()), sliding=int((risk == 1).sum()),
            toppling=int((risk == 2).sum()), drowning=int((risk == 3).sum()),
            affected_cells=int((risk > 0).sum()),
            affected_area_m2=int((risk > 0).sum() * C.DX * C.DY),
        ),
        max_depth=float(peak_h.max()),
        max_speed=float(peak_u.max()),
    )
    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(C.F_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    np.savez_compressed(C.OUT / "risk_map.npz", risk=risk, peak_h=peak_h, peak_u=peak_u)
    print(f"[L3] 输出: {C.F_METRICS.name}, risk_map.npz")


if __name__ == "__main__":
    main()
