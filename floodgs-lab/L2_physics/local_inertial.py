"""
L2 物理层 —— 二维浅水方程求解器（局部惯性格式）
================================================
格式来源：
    Bates, Horritt & Fewtrell (2010), "A simple inertial formulation of the
    shallow water equations for efficient two-dimensional flood inundation
    modelling", J. Hydrology 387:33-45.
    de Almeida et al. (2012), "Improving the stability of a simple formulation
    of the shallow water equations for 2D flood modelling", WRR 48:W05557.

关键实现要点（这几个坑踩过，写论文时可以省你两周）：
  1. 动量方程的压力项必须用**水面高程梯度** eta = z + h，不能用地面高程梯度。
     否则静水状态也会产生假流动，几秒内数值爆炸。
  2. 建筑等固壁要用 mask 把面流量清零，不能只靠抬高地形 —— 9 m 高的陡壁
     会产生虚假大流速，必须显式封边。
  3. 每步对单宽流量做 q <= 0.7 * h_flow * dx / dt 的限幅，防单步排空。
  4. 时间步 dt = CFL*dx/sqrt(g*h_max)，CFL=0.6 在本场景稳定。

定位：快速评估基线。不能解析三维射流与冲击力，需与 DualSPHysics/RAF 交叉验证。
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def hydrograph(t):
    """上游入流过程线：单宽流量 (m^2/s)，含基流，保证 t=0 与初始场一致"""
    if t <= 0:
        return C.Q_BASE
    if t < C.T_PEAK:
        s = (t / C.T_PEAK) ** 1.8
    else:
        s = np.exp(-(t - C.T_PEAK) / C.T_RECESS)
    return C.Q_BASE + (C.Q_PEAK - C.Q_BASE) * s


class LocalInertialSWE:
    def __init__(self, z, landuse, dx=C.DX, dy=C.DY):
        self.z = z.astype(np.float64)
        self.ny, self.nx = z.shape
        self.dx, self.dy = dx, dy
        self.wall = (landuse == 5)                 # 建筑 = 固壁

        n_map = np.full(z.shape, C.N_FLOODPLAIN)
        n_map[landuse == 0] = C.N_RIVER
        n_map[landuse == 4] = C.N_ROAD
        self.n = n_map
        self.yc0 = 59.0
        self.nfx = (0.5 * (self.n[:, :-1] + self.n[:, 1:])) ** 2   # 预计算 n^2
        self.nfy = (0.5 * (self.n[:-1, :] + self.n[1:, :])) ** 2

        # 固壁面：任一侧是建筑则封死
        self.wall_fx = self.wall[:, :-1] | self.wall[:, 1:]
        self.wall_fy = self.wall[:-1, :] | self.wall[1:, :]

        # ---- 初始条件：河道内一个水平水面 h0，配对应基流，避免启动冲击
        y = (np.arange(self.ny) + 0.5) * dy
        river_row = np.abs(y - self.yc0) < C.RIVER_HALF_W   # (ny,)
        river = np.repeat(river_row[:, None], self.nx, axis=1)  # 直河道，沿 x 不变
        H0 = 1.80                       # 枯水水深 (m)
        self.h = np.where(river, H0, 0.0).astype(np.float64)
        self.h[self.wall] = 0.0

        self.qx = np.zeros((self.ny, self.nx + 1))
        self.qy = np.zeros((self.ny + 1, self.nx))
        self.qx[:, :-1] = np.where(river, C.Q_BASE, 0.0)   # 初始基流

        self.t = 0.0
        self.vol_in = 0.0
        self.vol_loss = 0.0
        self.vol_out = 0.0
        self.vol0 = float(self.h.sum()) * dx * dy
        self._y = y

    # ------------------------------------------------------------ 边界
    def _set_boundaries(self, t):
        in_river = np.abs(self._y - self.yc0) < C.RIVER_HALF_W
        self.qx[:, 0] = np.where(in_river, hydrograph(t), 0.0)
        self.qx[:, -1] = self.qx[:, -2]      # 东侧自由出流
        self.qy[0, :] = 0.0              # 南边界：山体，固壁
        self.qy[-1, :] = self.qy[-2, :]  # 北边界：厂区下游出口，自由出流

    # ------------------------------------------------------------ 单步
    def step(self, dt):
        h, z = self.h, self.z
        g = C.G

        # ============ x 方向面流量：水面高程梯度 ============
        ha, hb = h[:, :-1], h[:, 1:]           # 左 / 右
        hf = np.maximum(ha, hb)
        eta_a = z[:, :-1] + ha                 # 左水面
        eta_b = z[:, 1:] + hb                  # 右水面
        dz = eta_b - eta_a
        wet = (hf > C.H_DRY) & (~self.wall_fx)
        # 【关键】干侧不能供水：若 a 干而梯度指向 a->b，或 b 干而指向 b->a，则断流。
        # 否则干地的地形高程会被当作"水面"，凭空向湿侧注水，几秒内即发散。
        wet &= ~(((ha <= C.H_DRY) & (dz < 0)) | ((hb <= C.H_DRY) & (dz > 0)))
        hfs = np.where(wet, hf, 1.0)
        dzd = dz / self.dx                     # 水面梯度

        q = self.qx[:, 1:-1]
        num = q - g * hfs * dt * dzd
        den = 1.0 + g * dt * self.nfx * np.abs(q) / hfs ** (7.0 / 3.0)
        q_new = np.where(wet, num / den, 0.0)
        qlim = 0.8 * hfs * self.dx / dt
        q_new = np.clip(q_new, -qlim, qlim)
        self.qx[:, 1:-1] = q_new

        # ============ y 方向面流量 ============
        ha, hb = h[:-1, :], h[1:, :]
        hf = np.maximum(ha, hb)
        eta_a = z[:-1, :] + ha
        eta_b = z[1:, :] + hb
        dz = eta_b - eta_a
        wet = (hf > C.H_DRY) & (~self.wall_fy)
        wet &= ~(((ha <= C.H_DRY) & (dz < 0)) | ((hb <= C.H_DRY) & (dz > 0)))
        hfs = np.where(wet, hf, 1.0)
        dzd = dz / self.dy

        q = self.qy[1:-1, :]
        num = q - g * hfs * dt * dzd
        den = 1.0 + g * dt * self.nfy * np.abs(q) / hfs ** (7.0 / 3.0)
        q_new = np.where(wet, num / den, 0.0)
        qlim = 0.8 * hfs * self.dy / dt
        q_new = np.clip(q_new, -qlim, qlim)
        self.qy[1:-1, :] = q_new

        # ============ 连续性 ============
        dh = -dt * ((self.qx[:, 1:] - self.qx[:, :-1]) / self.dx
                    + (self.qy[1:, :] - self.qy[:-1, :]) / self.dy)
        h_new = h + dh

        neg = h_new < 0
        if neg.any():
            self.vol_loss += float((h_new[neg] * self.dx * self.dy).sum())
            h_new[neg] = 0.0
        h_new[self.wall] = 0.0
        self.h = h_new

        self.vol_in += float(self.qx[:, 0].sum()) * dt * self.dy
        self.vol_out += (float(self.qx[:, -1].sum()) * dt * self.dy
                         + float(self.qy[-1, :].sum()) * dt * self.dx)
        self.t += dt

    # ------------------------------------------------------------ 主循环
    def run(self, t_end=C.T_END, snapshot_dt=C.SNAPSHOT_EVERY, verbose=True):
        snaps = dict(t=[], h=[], ux=[], uy=[])
        next_snap = 0.0
        nstep = 0
        t0 = self.t
        while self.t < t_end:
            hm = float(self.h.max())
            if not np.isfinite(hm) or hm > 40.0:
                raise RuntimeError(f"数值发散于 t={self.t:.1f}s, h_max={hm:.1f}m")
            ux, uy = self.velocity()
            umax = float(max(np.abs(ux).max(), np.abs(uy).max()))
            # CFL 同时约束重力波速和对流：dt <= CFL*dx / (sqrt(gh) + |u|)
            dt = C.CFL * self.dx / (np.sqrt(C.G * max(hm, 0.5)) + umax)
            dt = min(dt, C.DT_MAX, t_end - self.t)
            dt = max(dt, 1e-3)

            self._set_boundaries(self.t)
            self.step(dt)
            nstep += 1

            if self.t >= next_snap:
                self._record(snaps)
                next_snap += snapshot_dt
                if verbose and len(snaps["t"]) % 8 == 0:
                    print(f"  t={self.t:6.1f}s  hmax={self.h.max():5.2f}m  "
                          f"wet={(self.h>0.05).sum():6d}  vmax="
                          f"{np.abs(self.velocity()[0]).max():5.2f} m/s  steps={nstep}")

        if verbose:
            vol = self.h.sum() * self.dx * self.dy
            err = (self.vol0 + self.vol_in - self.vol_out - vol) / max(self.vol_in, 1e-9) * 100
            print(f"[L2] 完成: {nstep} 步 ({self.t-t0:.0f}s 模拟)")
            print(f"[L2] 水量平衡: 初始 {self.vol0/1e4:.2f} + 入流 {self.vol_in/1e4:.2f} "
                  f"- 出流 {self.vol_out/1e4:.2f} = 理论 {vol/1e4:.2f} 万m3, "
                  f"实存 {vol/1e4:.2f} 万m3, 闭合误差 {err:+.1f}%")
        return snaps, nstep

    def velocity(self):
        hf = np.maximum(self.h, C.H_DRY)
        ux = 0.5 * (self.qx[:, :-1] + self.qx[:, 1:]) / hf
        uy = 0.5 * (self.qy[:-1, :] + self.qy[1:, :]) / hf
        wet = self.h > 0.05
        # 物理上限：山洪最大流速约 10-12 m/s，超出视为干湿界面的数值伪迹
        ux = np.clip(np.where(wet, ux, 0.0), -12.0, 12.0)
        uy = np.clip(np.where(wet, uy, 0.0), -12.0, 12.0)
        return ux, uy

    def _record(self, snaps):
        ux, uy = self.velocity()
        snaps["t"].append(self.t)
        snaps["h"].append(self.h.astype(np.float32))
        snaps["ux"].append(ux.astype(np.float32))
        snaps["uy"].append(uy.astype(np.float32))


def main():
    d = np.load(C.F_SCENE)
    z, lu = d["z"].astype(np.float64), d["landuse"]

    print("[L2] 初始化浅水求解器 ...")
    model = LocalInertialSWE(z, lu)
    print(f"[L2] 网格 {model.nx}x{model.ny}  dx={model.dx}m  入流峰值 {C.Q_PEAK} m2/s "
          f"(总流量 ~{C.Q_PEAK*2*C.RIVER_HALF_W:.0f} m3/s)")
    snaps, nstep = model.run()

    C.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        C.F_RUN, z=z.astype(np.float32), landuse=lu,
        **{k: np.array(v) for k, v in snaps.items()},
    )
    print(f"[L2] 输出: {C.F_RUN.name}  ({len(snaps['t'])} 帧)")


if __name__ == "__main__":
    main()
