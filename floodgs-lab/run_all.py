"""
一键运行全流程
==============
    python run_all.py            # 完整流程（约 3 分钟）
    python run_all.py --fast     # 跳过渲染，只出数值结果（约 1.5 分钟）

各阶段可单独运行：
    python L1_scene/build_scene.py
    python L2_physics/local_inertial.py
    python L3_assess/assess.py
    python L4_render/render.py
    python make_figures.py
    python gen_viewer.py
"""

import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("L1 场景资产化", "L1_scene/build_scene.py", False),
    ("L2 浅水方程求解", "L2_physics/local_inertial.py", False),
    ("L3 灾害评估", "L3_assess/assess.py", False),
    ("L4 3DGS 渲染对比", "L4_render/render.py", True),
    ("L4 动态序列动画(3D)", "L4_render/animate.py", True),
    ("L4 动态序列动画(2D)", "L4_render/plot_2d.py", True),
    ("    配图生成", "make_figures.py", True),
    ("    看板生成", "gen_viewer.py", True),
]


def main():
    fast = "--fast" in sys.argv
    t_all = time.time()
    for name, script, render_step in STEPS:
        if fast and render_step:
            print(f"\n>>> [跳过] {name}")
            continue
        print(f"\n{'='*62}\n>>> {name}\n{'='*62}")
        t = time.time()
        r = subprocess.run([PY, str(PROJ / script)], cwd=str(PROJ))
        if r.returncode != 0:
            print(f"\n!! {name} 失败，退出码 {r.returncode}")
            return 1
        print(f"<<< {name} 完成，用时 {time.time()-t:.1f}s")

    print(f"\n{'='*62}")
    print(f"全流程完成，总用时 {time.time()-t_all:.1f}s")
    print(f"看板: {PROJ/'viewer'/'index.html'}")
    print(f"对比图: {PROJ/'outputs'/'compare_before_after.png'}")
    print(f"动画:   {PROJ/'outputs'/'flood_impact.mp4'}")
    print(f"指标: {PROJ/'outputs'/'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
