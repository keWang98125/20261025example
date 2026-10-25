# floodgs-lab

面向 3DGS 场景的洪水冲击灾害仿真**最小可运行闭环**。
四层解耦：场景资产化 → 浅水物理 → 状态级评估 → 灾前/灾后对比。

**全部 CPU 运行，不依赖 GPU。** 全流程约 110 秒，内存峰值 < 2 GB。

```
L1 场景资产化   build_scene.py        地形 + 语义 + 物体清单（对齐 InsertAny3D 的 pose.json）
L2 物理仿真     local_inertial.py     二维浅水方程（局部惯性格式，LISFLOOD-FP 同源）
L3 灾害评估     assess.py             车辆失稳 + 货物散开 + 人体三模式风险栅格
L4 渲染对比     render.py             numpy 实现的 3DGS 光栅化器，灾前/灾后同视角
```

---

## 1. 快速开始

```bash
cd floodgs-lab
.venv/Scripts/python.exe run_all.py          # Windows
# 或 python run_all.py
```

产物：

| 文件 | 内容 |
|---|---|
| `viewer/index.html` | 自包含看板（双击打开，图片已内嵌） |
| `outputs/compare_before_after.png` | 灾前/灾后 3DGS 对比图 |
| `outputs/metrics.json` | 全部评估指标 |
| `outputs/fig_*.png` | 论文用图（框架/过程线/水深/风险/轨迹） |
| `outputs/flood_impact.mp4` | **3D 动态序列**：0–600 s 全景，看整体淹没过程 |
| `outputs/flood_impact_2d.mp4` | **2D 俯视特写**：0–500 s，水深+流速+物体位姿，看细节最清楚 |
| `outputs/frames/` | 动画序列帧（JPEG，可自行拼视频或插入 PPT） |

单阶段运行：

```bash
python L1_scene/build_scene.py      # 1 s
python L2_physics/local_inertial.py # 80 s
python L3_assess/assess.py          # 19 s
python L4_render/render.py          # 5 s
python L4_render/animate.py         # 约 2.5 min，3D 全景动画
python L4_render/plot_2d.py         # 约 20 s，2D 俯视特写动画
```

动画参数：

```bash
python L4_render/animate.py --dt 5 --tend 600 --fps 12   # 默认
python L4_render/animate.py --dt 2 --tend 350 --fps 20   # 冲击段慢放（更细）
python L4_render/animate.py --scale 1.5                  # 提高分辨率（更慢）
python L4_render/animate.py --cam close                  # 近景机位（地形高斯尺度不匹配，会偏糊）
python L4_render/plot_2d.py --tend 350 --dt 2 --fps 20   # 冲击段慢放
```

> 为什么近景用 2D 而不是拉近 3D：地形高斯按 2 m 间隔采样，相机拉近到 40 m 时
> 单个高斯直径就占 40+ 像素，画面会糊成一团。2D 俯视没有这个问题，信息密度更高。

### 动画的两种定位

| 版本 | 用途 | 特点 |
|---|---|---|
| 3D 全景 | 答辩/汇报，直观感受 | 有体积感，物体细节弱 |
| 2D 特写 | 论文配图、细节论证 | 水深/流速/位姿全部可读 |

动画是怎么做的（不是后期剪辑）：静态地形/建筑点云只构建一次，每帧从
`sim_result.npz` 插值水深场、从 `metrics.json` 插值物体轨迹，重建动态高斯后重新光栅化。
换场景或改参数，重跑就得到新视频。

---

## 2. 这个场景是什么

一条自西向东的山区河道（22 m 宽、堤顶高 4.2 m），北岸是低洼厂区（地面比堤顶低 1.8 m）。
上游山洪（单宽峰值 20 m²/s ≈ 440 m³/s）→ 堤防在 x=110 m 处有 22 m 宽溃口 →
洪水涌入厂区形成 0.4~1.1 m 的过水片流，向北排出。

算例里放了 6 辆车、54 箱货物、11 个人员，用于演示三类评估。

> ⚠️ 这是**参数化合成场景**，不是你研究区的真实地形。
> 它的作用是验证方法链路，论文里的实际结果必须换成真实数据（见第 4 节）。

---

## 3. 为什么这么设计（写给第一次写论文的你）

### 3.1 不要试图"重建物理场"，要"插入成熟模块"

你老师说的"添加新物体/3D Mesh 已经有很多研究了"是关键提示。
本仓库的定位就是：**不发明物理引擎，只做缝合层**。

- L2 的浅水求解器只用了 200 行，因为它本来就该这么短 —— 局部惯性格式是洪水淹没模拟的标准工具
- 但它**不是**高保真求解器：不能解析三维射流、不能算冲击力
- 所以论文里必须把它定位为"快速评估基线"，并用 DualSPHysics 或 RAF 做交叉验证

### 3.2 三个创新点的诚实表述

| 你想说的 | 建议改成这样说 | 原因 |
|---|---|---|
| "首次将 3DGS 与物理引擎结合" | "提出一套 3DGS 场景资产到多求解器后端的桥接接口，并在灾害冲击场景中验证" | "首次"很容易被审稿人打掉 |
| "复现了 SWE-DEM 全耦合" | "采用基于浮力-拖曳平衡的简化刚体判据，与 Xiong et al. (2024) 的 DEM 结果做趋势一致性对比" | 你跑的不是 DEM |
| "提出了新的人体失稳模型" | "复现并应用 Milanesi et al. (2015) 的三模式判据，生成场景级风险栅格" | 模型是别人的 |

### 3.3 为什么评估层要用"状态"而不是"动画"

老师说"汽车被冲到河里、货物被冲散、人员被淹"不该做成连续动画 —— 这是对的。
动画不可验证。状态标签可以：

- 车辆：失稳/未失稳（二分类）+ 输运终点误差（回归）
- 货物：散开半径比（RMS 半径）
- 人员：四类风险等级（安全/滑移/倾覆/溺水）

每一个都能和文献值、现场调查、或高保真模型对照。

---

## 4. 换成你自己的研究区

### 4.1 替换地形

只需要提供 `assets/scene.npz`，字段：

| 字段 | 形状 | 说明 |
|---|---|---|
| `z` | (ny, nx) float32 | 地面高程 (m)，含建筑 footprint 抬高 |
| `landuse` | (ny, nx) uint8 | 0 河道 1 堤防 2 厂区 3 山坡 4 道路 5 建筑 |
| `xc`, `yc` | (nx,), (ny,) | 单元中心坐标 |
| `vehicles` | (N, 4) | x, y, heading, z |
| `cargo` | (M, 4) | x, y, z, pile_id |
| `humans` | (K, 3) | x, y, z |

从真实 DEM（GeoTIFF）转换的模板：

```python
import numpy as np, rasterio
from scipy.ndimage import zoom

src = rasterio.open("your_dem.tif")
z = src.read(1).astype(np.float32)
z[z < -9999] = np.nanmedian(z)

# 重采样到 config.py 里的 NX/NY
f = (200 / z.shape[0], 200 / z.shape[1])
z = zoom(z, f, order=1)

# 流向：确认 DEM 的 y 方向（rasterio 默认北在上，本仓库 y 向北为正）
# 若栅格是 north-up，需要翻转：
z = z[::-1, :]

np.savez_compressed("assets/scene.npz", z=z, landuse=np.full(z.shape, 2, np.uint8), ...)
```

改 `config.py` 的 `NX/NY/DX/DY` 时注意：`DX = 域宽 / NX`，保持真实尺度。

### 4.2 标定入流

真实流域的洪峰流量用推理公式或水文站资料：

```
Q_peak (m³/s)  →  q_peak = Q_peak / 河宽
T_PEAK: 流域汇流时间
```

改 `config.py` 的 `Q_PEAK`、`T_PEAK`、`T_RECESS`。
**验证方法**：跑出的主槽水位应与实测洪痕或 HEC-RAS 结果差 < 0.3 m。

---

## 5. 参数在哪

全部集中在 `config.py`：

| 参数 | 含义 | 本文取值 |
|---|---|---|
| `Q_PEAK` | 上游单宽流量峰值 | 20 m²/s |
| `BED_SLOPE` | 河底纵坡 | 0.008 |
| `YARD_OFFSET` | 厂区相对河底高差 | 2.4 m |
| `LEVEE_TOP_OFFSET` | 堤顶相对河底高差 | 4.2 m |
| `N_FLOODPLAIN` / `N_RIVER` | Manning 糙率 | 0.055 / 0.035 |
| `CFL` | 时间步系数 | 0.45 |
| `VEHICLE_TYPES` | 车辆质量/尺寸/阻力系数 | 见文件 |

---

## 6. 已知限制（论文局限一节直接抄）

1. **单向耦合**：水流驱动刚体，刚体不反作用于流场。真实中车辆会阻水、改变局部流态。
2. **浅水假设**：忽略垂向加速度，溃口射流的垂直结构无法解析；峰值流速可能被低估或高估。
3. **刚体简化**：车辆按质点 + 姿态处理，无 DEM 接触、无翻滚。
4. **合成场景**：地形与物体布置为参数化生成，不是实测数据。
5. **干湿前沿**：局部惯性格式在薄水层（< 0.1 m）上存在可见数值震荡，影响流速瞬时值（水深较可靠）。
6. **人体模型**：复现了力矩平衡思路，参数取自论文正文，未逐项核对原作者代码。

---

## 7. 环境

```
Python 3.13
numpy 2.5.4 / scipy 1.18 / matplotlib 3.11 / pillow
```

虚拟环境已建在 `.venv/`。重装：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy scipy matplotlib pillow -i https://pypi.org/simple
```

> 注意：本机 pip 默认源配置指向清华镜像但不可用，命令里要显式加 `-i https://pypi.org/simple`。

---

## 8. 下一步做什么

见 `docs/实验方案.md`（论文实验设计）和 `docs/6GB部署指南.md`（算力分级方案）。
