# 6 GB 显存档位下的部署指南

你的机器：**RTX 3060 Laptop 6 GB（桌面占用后实际可用约 4.5 GB）/ 16 GB RAM / Ryzen 7 5800H 8C16T**

先说结论：**显存不是最紧的，内存才是。** 16 GB 内存在跑 3DGS 训练时会先于显存耗尽。

---

## 1. 分级：什么能跑，什么别硬来

| 模块 | 结论 | 需要的资源 | 备注 |
|---|---|---|---|
| 本仓库全流程 | ✅ 完全够 | CPU 1 核 + 1 GB 内存 | 110 s 跑完 |
| DualSPHysics（CPU 版） | ✅ 可以 | 8 核 + 4 GB | 20 万粒子以内 |
| gsplat 小场景训练 | 🟡 勉强 | 4.5 GB 显存 + 12 GB 内存 | 800×450、< 100 张图、点云 < 30 万 |
| 官方 3DGS（diff-gaussian-rasterization） | 🟡 勉强 | 同上，更容易 OOM | 必须 `-r 2` 降采样 |
| Gaussian Grouping | 🟡 勉强 | 同 3DGS + SAM | 建议先降分辨率训 3DGS 再加载 |
| SAP / SAM2 分割 | ✅ 可以 | 2-3 GB 显存 | 单图推理没问题 |
| InsertAny3D | ❌ 别本地跑 | SD + 多视角渲染，> 10 GB | 用替代方案（见 §5） |
| RAF / Genesis | ❌ 别本地跑 | 官方建议 ≥ 24 GB | 服务器或云 |
| OpenFOAM + LIGGGHTS | ❌ | 内存 > 32 GB | 服务器 |

---

## 2. 本机训练 3DGS 的具体做法（如果你非要试）

### 2.1 环境（不要用本仓库的 .venv，单独建）

```bash
# CUDA 12.1 版 PyTorch
python -m venv .venv-cuda
.venv-cuda/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3DGS 官方
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install plyfile tqdm
```

### 2.2 训练参数（6 GB 的关键）

```bash
# 1) 先把图像降采样，这是最有效的一招
python train.py -s data/your_scene -r 2 -m output/run1 \
    --iterations 15000 \
    --densify_until_iter 8000 \
    --densify_grad_threshold 0.0004 \
    --max_sh_degree 2 \
    --data_device cpu
```

关键参数解释：

| 参数 | 作用 | 为什么要这么设 |
|---|---|---|
| `-r 2` | 图像降采样 2 倍 | 显存占用约降 75% |
| `--data_device cpu` | 图像放 CPU 内存 | 省显存，但需要足够内存 |
| `--max_sh_degree 2` | 球谐降到 2 阶 | 每点参数从 48 降到 12 |
| `--densify_until_iter 8000` | 提前停止增点 | 控制点云规模 < 30 万 |
| `--iterations 15000` | 迭代数 | 7k 太少质量差，30k 会 OOM |

### 2.3 必设的环境变量

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

不加这个，6 GB 下会频繁碎片化 OOM。

### 2.4 监控

训练时另开一个窗口：

```bash
nvidia-smi -l 1
```

显存 > 5.6 GB 就说明随时会炸。

如果 OOM，按顺序降级：
1. `-r 2` → `-r 4`
2. `--iterations 15000` → `10000`
3. `--densify_grad_threshold` 调大到 `0.0008`（少增点）
4. 关掉所有浏览器（Edge 能占 1.5 GB 显存）

---

## 3. DualSPHysics（这台机器上最好用的高保真后端）

**它是纯 CPU 求解器**，不吃显存，8 核 16 线程正好合适。

```bash
# 1) 下载 Windows 预编译版
#    https://dual.sphysics.org/  → Download → DualSPHysics_v5.x_Windows64.zip
# 2) 解压到 D:\DualSPHysics
# 3) 用 GenCase 生成算例，DualSPHysics5.x_win64.exe 求解
```

跑一个溃口射流子区域的推荐配置：

```
域尺寸：40 m × 30 m × 8 m
粒子间距 dp = 0.10 m  →  约 96 万粒子（8 核跑 6-12 小时）
粒子间距 dp = 0.15 m  →  约 28 万粒子（8 核跑 1-2 小时）  ← 推荐先跑这个
```

输出 `Part_*.bi4`，用 Paraview 打开，或者写脚本导出指定位置的水深/流速时程，
再和本仓库的浅水结果对比。

> 论文写法：不要声称 DualSPHysics 结果就是"真值"，写"高保真参考解（high-fidelity reference）"。

---

## 4. 云 GPU 怎么用最省钱

国内推荐 **AutoDL / 恒源云**（按小时计费，学生有优惠）：

| 卡型 | 显存 | 价格（约） | 适合 |
|---|---|---|---|
| RTX 3090 | 24 GB | ¥1.3-1.8/h | 3DGS 训练 |
| RTX 4090 | 24 GB | ¥2.0-2.6/h | 3DGS 训练 + InsertAny3D |
| A100 40G | 40 GB | ¥5-8/h | RAF / Genesis |

**预算估算**：整个论文的重活大约是

- 3DGS 重建 3 个场景：每个 1-2 h → 约 ¥15
- Gaussian Grouping 分割：每个 1 h → 约 ¥5
- InsertAny3D 试跑：3-5 h → 约 ¥15
- RAF/Genesis 复现：5 h → 约 ¥30

**合计 100 元以内能跑完全部实验。** 比升级显卡划算得多。

建议流程：
1. 本地把代码调通（用本仓库这套 CPU 闭环验证逻辑）
2. 数据打包上传
3. 云上跑重活，只下载结果（.ply / 图 / 指标 json）
4. 本地做分析和写作

---

## 5. InsertAny3D 的轻量替代（重要）

InsertAny3D 需要 Stable Diffusion + 多视角渲染，6 GB 跑不动，云上也很折腾。
但它的核心功能只有一个：**把物体放进 3DGS 场景并估计位姿**。

替代方案（成功率更高、成本更低）：

**方案 A：Blender 生成 + 手动配准**
1. Blender 里建一个车辆模型（或从 free3d 下载）
2. 用 `gsplat` 对单物体渲染 100 张图 → 训练成高斯
3. 在 3DGS 场景的 viewer 里手动拖到目标位置，记录 6DoF
4. 输出 `inserted_object.ply` + `pose.json`（格式与本仓库 `scene_asset.json` 一致）

精度足够（灾害评估不需要毫米级位姿），工作量 1-2 天。

**方案 B：直接用 3DGS 场景里的真实车辆**
如果你的重建场景里本来就有车，用 Gaussian Grouping 分割出来，直接作为动态资产 ——
不需要插入任何东西。这最省事，也最真实。

**方案 C：SAP（Segment Any 3D Gaussians）**
比 Gaussian Grouping 轻，分割质量略低但够用。

> 论文里写"我们采用物体级资产插入/分割流程"，不要写死某个具体工具名，
> 这样审稿人问起来你还有替换空间。

---

## 6. 常见报错

| 报错 | 原因 | 解决 |
|---|---|---|
| `CUDA out of memory` | 显存不足 | 加 `-r 2`、降迭代、关浏览器 |
| `No module named diff_gaussian_rasterization` | 子模块没装 | `pip install submodules/diff-gaussian-rasterization` |
| `nvcc not found` | CUDA toolkit 未装 | 装 CUDA 12.1 toolkit，或改用 gsplat（纯 Python 编译） |
| `Killed` / 系统卡死 | **内存不足**（不是显存） | 加 swap、降 `--data_device cpu` 的图数 |
| `AssertionError: image size mismatch` | `-r` 与已生成的 cache 不一致 | 删掉 `output/*/` 重跑 |
| pip 装不上包 | 本机默认源不可用 | 加 `-i https://pypi.org/simple` |

---

## 7. 一句话策略

**本机负责"想清楚"和"验证逻辑"，云上负责"跑重活"。**

不要为了在本地跑通 InsertAny3D 折腾一周 —— 那一周够你把实验方案做完一半了。
