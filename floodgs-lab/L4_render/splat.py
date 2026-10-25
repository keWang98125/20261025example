"""
L4-2 轻量 3D 高斯光栅化器（纯 numpy，无需 GPU）
================================================
实现 3DGS 前向渲染的完整链路，与官方 diff-gaussian-rasterizer 的数学一致，
只是去掉了 CUDA tile 调度：
    1. 世界坐标 -> 相机坐标（look-at）
    2. 3D 协方差投影到 2D（EWA splatting 的 Jacobian 近似）
    3. 按深度降序排序
    4. 逐高斯在包围盒内做 alpha 混合

性能：约 2.5 万个高斯 / 720x540 分辨率，单帧 1~3 s（Ryzen 7 5800H）。
用途：本地快速出图、论文示意图；不用于训练。
"""

import numpy as np


def look_at(eye, target, up=(0, 0, 1)):
    """返回 world->camera 的 R, t；相机看向 +z，图像 y 向下"""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    up = np.asarray(up, float)
    zc = target - eye; zc /= np.linalg.norm(zc)        # 视线方向 = 相机 +z
    xc = np.cross(zc, up); xc /= np.linalg.norm(xc)    # 图像 +x（向右）
    yc = np.cross(xc, zc)                              # 图像 +y（向上）
    R = np.stack([xc, yc, zc])                          # world -> camera
    t = -R @ eye
    return R, t


def project_and_render(means, colors, scales, opacities, width, height,
                       eye, target, fov_deg=55.0, bg=(0.93, 0.95, 0.97),
                       near=1.0, max_sigma=3.0, white_bg=True):
    R, t = look_at(eye, target)
    P = means @ R.T + t                                 # (N,3) 相机坐标
    z = P[:, 2]
    valid = z > max(near, 5.0)
    P, z = P[valid], z[valid]
    colors = colors[valid]; scales = scales[valid]; opacities = opacities[valid]
    if len(z) == 0:
        return np.tile(np.array(bg), (height, width, 1))

    f = 0.5 * height / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0

    u = f * P[:, 0] / z + cx
    v = -f * P[:, 1] / z + cy                           # 图像 y 向下

    # ---- 2D 协方差：Sigma2d = J W Sigma W^T J^T，球协方差 sig^2 I 时化简
    sig2 = (scales ** 2)
    a = f / z
    b = f * P[:, 0] / (z ** 2)
    c = f * P[:, 1] / (z ** 2)
    # J = [[a,0,-b],[0,-a,c]]  (y 取负后行号对应)
    s11 = sig2 * (a ** 2 + b ** 2)
    s12 = sig2 * (-b * c)
    s22 = sig2 * (a ** 2 + c ** 2)
    # 膨胀 1 px 抗锯齿
    s11 += 0.35; s22 += 0.35

    det = s11 * s22 - s12 ** 2
    det = np.maximum(det, 1e-9)
    inv11 = s22 / det; inv12 = -s12 / det; inv22 = s11 / det

    # 3-sigma 包围盒
    rad = max_sigma * np.sqrt(np.maximum(s11, s22))
    x0 = np.floor(u - rad).astype(int); x1 = np.ceil(u + rad).astype(int)
    y0 = np.floor(v - rad).astype(int); y1 = np.ceil(v + rad).astype(int)

    inside = (x1 >= 0) & (x0 < width) & (y1 >= 0) & (y0 < height)
    order = np.argsort(-z[inside])                       # 远 -> 近
    idx_all = np.nonzero(inside)[0][order]

    img = np.tile(np.array(bg, dtype=np.float32), (height, width, 1))

    u_f, v_f = u, v
    for k in idx_all:
        xi0 = max(x0[k], 0); xi1 = min(x1[k], width - 1)
        yi0 = max(y0[k], 0); yi1 = min(y1[k], height - 1)
        if xi1 < xi0 or yi1 < yi0:
            continue
        xs = np.arange(xi0, xi1 + 1); ys = np.arange(yi0, yi1 + 1)
        dx = xs[None, :] - u_f[k]
        dy = ys[:, None] - v_f[k]
        power = -0.5 * (inv11[k] * dx * dx + 2 * inv12[k] * dx * dy + inv22[k] * dy * dy)
        alpha = opacities[k] * np.exp(power)
        alpha = np.clip(alpha, 0.0, 0.99)
        if alpha.max() < 1e-4:
            continue
        # painter 算法（远->近）：C = c*a + C*(1-a)
        a3 = alpha[..., None]
        patch = img[yi0:yi1 + 1, xi0:xi1 + 1]
        img[yi0:yi1 + 1, xi0:xi1 + 1] = a3 * colors[k] + patch * (1.0 - a3)

    return np.clip(img, 0, 1)
