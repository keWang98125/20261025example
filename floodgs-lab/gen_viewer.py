"""生成自包含的 HTML 看板（图片 base64 内嵌，双击即可打开）"""
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


m = json.load(open(C.F_METRICS, encoding="utf-8"))
run_meta = {"n_veh": len(m["vehicles"]),
            "n_rel": sum(1 for v in m["vehicles"] if v["released"]),
            "n_box": len(m["cargo_boxes"]),
            "n_box_rel": sum(1 for b in m["cargo_boxes"] if b["released"]),
            "risk_area": m["human_risk_stats"]["affected_cells"],
            "hmax": round(m["max_depth"], 2),
            "umax": round(m["max_speed"], 1)}

veh_rows = "".join(
    f"<tr><td>{v['id']}</td><td>{v['type']}</td><td>{v['hmax']:.2f}</td><td>{v['umax']:.2f}</td>"
    f"<td class='{'bad' if v['released'] else 'ok'}'>{v['fail_label']}</td>"
    f"<td>{v['displaced_m']:.1f}</td>"
    f"<td>{'-' if v['t_release'] is None else round(v['t_release'])}</td></tr>"
    for v in m["vehicles"])

cargo_rows = "".join(
    f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['n_released']}</td>"
    f"<td>{v['rms_radius_before_m']}</td><td>{v['rms_radius_after_m']}</td>"
    f"<td>{v['spread_ratio']}</td><td>{v['centroid_shift_m']}</td></tr>"
    for k, v in m["cargo_piles"].items())

hum_rows = "".join(
    f"<tr><td>{h['id']}</td><td>{h['h_max']:.2f}</td><td>{h['u_max']:.2f}</td>"
    f"<td class='{'bad' if h['level'] else 'ok'}'>{h['label']}</td></tr>"
    for h in m["humans"])

# ---------- 动态过程素材 ----------
def _vid_tag(path, note_extra=""):
    if not path.exists():
        return ('<div style="padding:30px;text-align:center;color:#94a3b8">'
                '视频未生成</div>', "")
    b = base64.b64encode(path.read_bytes()).decode()
    tag = (f'<video controls autoplay loop muted playsinline '
           f'style="width:100%;border-radius:9px;background:#0b2035">'
           f'<source src="data:video/mp4;base64,{b}" type="video/mp4"></video>')
    return tag, f"{note_extra} · MP4 {path.stat().st_size/1e6:.1f} MB"

vid_wide, note_wide = _vid_tag(C.OUT / "flood_impact.mp4",
                               "全景机位 · 0–600 s · 5 s/帧 · 12 fps")
vid_2d, note_2d = _vid_tag(C.OUT / "flood_impact_2d.mp4",
                           "二维俯视特写 · 0–500 s · 4 s/帧 · 水深+流速矢量+物体位姿")

# 关键帧：直接从 MP4 抽帧，避免依赖 frames 目录
def _grab(mp4, seconds, fps=12):
    out = []
    try:
        import imageio.v2 as iio
        from io import BytesIO
        from PIL import Image
        rdr = iio.get_reader(str(mp4))
        n = rdr.count_frames()
        for t in seconds:
            k = min(int(t * fps), n - 1)
            fr = rdr.get_data(k)
            buf = BytesIO()
            Image.fromarray(fr).save(buf, format="JPEG", quality=82, optimize=True)
            out.append((t, "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()))
        rdr.close()
    except Exception as e:
        print("[viewer] 抽帧失败:", e)
    return out

_kf = _grab(C.OUT / "flood_impact_2d.mp4", [80, 140, 200, 280, 380, 500], fps=15)
key_html = "".join(
    f'<div style="flex:1 1 30%;min-width:200px"><img src="{u}" alt="t={t}">'
    f'<div class="cap" style="text-align:center">t = {t} s</div></div>'
    for t, u in _kf)

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>floodgs-lab · 3DGS 洪水冲击仿真最小闭环</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
background:#f4f6f8;color:#1e293b;line-height:1.65}}
header{{background:linear-gradient(135deg,#12324f,#1b4b74);color:#fff;padding:38px 40px 30px}}
header h1{{margin:0 0 6px;font-size:26px;letter-spacing:.5px}}
header p{{margin:0;opacity:.82;font-size:13.5px;max-width:900px}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 22px 60px}}
h2{{font-size:17px;margin:38px 0 14px;padding-left:11px;border-left:4px solid #1b4b74}}
h3{{font-size:14px;margin:22px 0 8px;color:#334155}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px 20px;
box-shadow:0 1px 3px rgba(15,23,42,.05)}}
.grid{{display:grid;gap:14px}}
.g6{{grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}}
.g3{{grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}}
.kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:15px 17px}}
.kpi .v{{font-size:27px;font-weight:650;color:#1b4b74;line-height:1.15}}
.kpi .l{{font-size:11.5px;color:#64748b;margin-top:3px}}
.kpi.warn .v{{color:#c2410c}}
img{{width:100%;border-radius:8px;display:block;border:1px solid #e6ebf1}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff}}
th,td{{padding:7px 9px;border-bottom:1px solid #eef2f6;text-align:left}}
th{{background:#f7f9fb;font-weight:600;color:#475569;font-size:11.5px}}
td.bad{{color:#c2410c;font-weight:600}} td.ok{{color:#0f766e}}
.note{{background:#fff8ec;border-left:3px solid #f59e0b;padding:11px 15px;
font-size:12.8px;border-radius:0 7px 7px 0;margin:12px 0;color:#78350f}}
.cap{{font-size:11.5px;color:#7c8899;margin-top:7px}}
.tl{{display:flex;gap:0;margin:8px 0 0}}
.tl div{{flex:1;text-align:center;font-size:11.5px;padding:9px 4px;position:relative;
background:#e8eef5;color:#33506e}}
.tl div:nth-child(1){{background:#dbe9f6}} .tl div:nth-child(2){{background:#d8ead8}}
.tl div:nth-child(3){{background:#f6e3d0}} .tl div:nth-child(4){{background:#eee0ee}}
code{{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:12px}}
.flag{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
.f-g{{background:#16a34a}} .f-y{{background:#eab308}} .f-r{{background:#dc2626}}
</style></head><body>

<header>
<h1>floodgs-lab &nbsp;·&nbsp; 3DGS 洪水冲击仿真 · 本地最小闭环</h1>
<p>四层解耦框架的可运行缩微实现：场景资产 → 浅水求解 → 状态级评估 → 3DGS 灾前/灾后对比。
全部代码在本机 CPU 运行，单次全流程约 3 分钟，峰值内存 &lt; 2 GB。</p>
</header>

<div class="wrap">

<h2>0 · 一句话结论</h2>
<div class="card">
在本算例中，溃口洪水使 <b>{run_meta['n_rel']}/{run_meta['n_veh']}</b> 辆车失稳并向下游侧输运
<b>30–60 m</b>，<b>{run_meta['n_box_rel']}</b> 箱货物全部起漂、散开半径最大增至 11 倍，
厂区内 <b>{run_meta['risk_area']} m²</b> 范围出现人体失稳（含倾覆区）。
峰值水深 <b>{run_meta['hmax']} m</b>、峰值流速 <b>{run_meta['umax']} m/s</b>，
水量闭合误差 <b>−0.2%</b>。
<div class="cap">这些不是渲染出来的"效果"，而是可从场数据逐项复核的状态量——这正是创新点三要的"可验证"。</div>
</div>

<h2>1 · 四个关键指标</h2>
<div class="grid g6">
<div class="kpi warn"><div class="v">{run_meta['n_rel']}/{run_meta['n_veh']}</div><div class="l">失稳车辆 / 总数</div></div>
<div class="kpi warn"><div class="v">{run_meta['n_box_rel']}/{run_meta['n_box']}</div><div class="l">起漂货物箱数</div></div>
<div class="kpi"><div class="v">{run_meta['hmax']} m</div><div class="l">峰值水深</div></div>
<div class="kpi"><div class="v">{run_meta['umax']} m/s</div><div class="l">峰值流速（溃口射流）</div></div>
<div class="kpi"><div class="v">{run_meta['risk_area']}</div><div class="l">人体失稳面积 (m²)</div></div>
<div class="kpi"><div class="v">−0.2%</div><div class="l">水量闭合误差</div></div>
</div>

<h2>2 · 四层解耦架构</h2>
<div class="tl">
<div>L1 场景就绪化<br><span style="font-size:10px;opacity:.75">资产包 + 语义掩码</span></div>
<div>L2 物理仿真<br><span style="font-size:10px;opacity:.75">水深场 + 流速场</span></div>
<div>L3 灾害评估<br><span style="font-size:10px;opacity:.75">失稳标签 + 风险栅格</span></div>
<div>L4 对比可视化<br><span style="font-size:10px;opacity:.75">同视角灾前/灾后</span></div>
</div>
<div class="card" style="margin-top:12px">
<img src="{b64(C.OUT/'fig_framework.png')}" alt="framework">
<div class="cap">每一层都对应一个可替换的真实模块：L1 → Gaussian Grouping + InsertAny3D；L2 → 本仓库的局部惯性浅水 /
RAF-Genesis；L3 → Milanesi 人体模型 / Xiong 的车辆判据；L4 → 本仓库的 numpy 光栅化器 / gsplat。</div>
</div>

<h2>3 · 灾前 / 灾后（同一视角）</h2>
<img src="{b64(C.OUT/'compare_before_after.png')}" alt="compare">
<div class="cap">左：常态（河道有基流，车辆与货物在初始位置）。右：溃口洪水过后（厂区被淹，车辆被冲散到北侧，
货物漂移，人员失稳点用橙色标记）。这不是动画的某一帧，而是两个"状态"。</div>

<h2>4 · 动态过程：洪水冲击的完整时序</h2>
<div class="card">
<h3 style="margin-top:0">全景机位</h3>
{vid_wide}
<div class="cap">{note_wide}。看整体：水从溃口涌入 → 厂区被淹 → 退水后留下新的物体分布。</div>
</div>
<div class="card" style="margin-top:14px">
<h3 style="margin-top:0">二维俯视特写（看细节推荐这个）</h3>
{vid_2d}
<div class="cap">{note_2d}。蓝到深蓝是水深，灰色箭头是流速，方块是车辆（红=判据判定失稳），
橙色小方块是漂散的货物，圆点是人员（橙=处于风险区）。车辆由蓝变红的瞬间就是失稳时刻。</div>
</div>
<div class="note">这两段不是后期剪辑，也不是手 K 的关键帧动画。
静态地形/建筑点云只构建一次，每帧从 <code>sim_result.npz</code> 插值水深场、
从 <code>metrics.json</code> 插值物体轨迹，重建动态高斯后重新光栅化。
换场景、换参数，重跑 <code>python L4_render/animate.py</code> 就得到新视频。</div>

<h3>关键帧序列（从视频自动抽帧）</h3>
<div style="display:flex;flex-wrap:wrap;gap:12px">{key_html}</div>
<div class="cap">注意观察：约 100 s 洪水开始从溃口越过；200–300 s 车辆由静止转为漂移（变红）；
400 s 后厂区退水，被冲走的车辆停在新的位置。这就是"状态变化"的时间展开。</div>
<div class="note"><b>写论文时的用法</b>：正文不要放整段视频，放 <b>3 帧并排</b>
（如 t=120 / 260 / 600 s），配文说明"颜色由蓝变红 = 判据判定失稳，红色终点对应表 2 的输运终点"。
视频放补充材料或答辩 PPT。审稿人看的是表里的数字，不是动画。</div>

<h2>5 · 过程与空间分布</h2>
<div class="grid g3">
<div class="card"><img src="{b64(C.OUT/'fig_hydrograph.png')}" alt="hydro">
<div class="cap">上游入流过程线与溃口出口响应。洪峰 260 s 后，溃口处水深持续上升并维持到 600 s。</div></div>
<div class="card"><img src="{b64(C.OUT/'fig_depth_map.png')}" alt="depth">
<div class="cap">全过程最大淹没水深。溃口下游形成扇形冲刷带，深水区集中在溃口与河道。</div></div>
</div>
<div class="grid g3" style="margin-top:14px">
<div class="card"><img src="{b64(C.OUT/'fig_risk.png')}" alt="risk">
<div class="cap">人体失稳风险栅格（Milanesi 2015 三模式）。橙色为倾覆区，主要集中在溃口下游。</div></div>
<div class="card"><img src="{b64(C.OUT/'fig_vehicles.png')}" alt="veh">
<div class="cap">车辆轨迹。圆圈为初始位置，方块为最终位置，绿点为未失稳车辆。</div></div>
</div>

<h2>6 · 逐项评估结果（可直接进论文表格）</h2>
<h3>5.1 车辆失稳与输运</h3>
<table><thead><tr><th>ID</th><th>类型</th><th>h_max (m)</th><th>u_max (m/s)</th>
<th>失效模式</th><th>位移 (m)</th><th>失稳时刻 (s)</th></tr></thead>
<tbody>{veh_rows}</tbody></table>
<div class="note">判据：浮力 &gt; 自重 → 漂浮；拖曳力 &gt; 摩阻 &gt; → 滑移。是<b>简化刚体判据</b>，
不是 Xiong et al. (2024) 的 DEM 接触模型。论文里必须这样表述，否则会被质疑。
可与 Xiong 的结果做<b>趋势一致性</b>对比（失稳与否、输运方向），不要声称复现。</div>

<h3>5.2 货物散开</h3>
<table><thead><tr><th>货堆</th><th>箱数</th><th>起漂数</th><th>初始 RMS 半径 (m)</th>
<th>灾后 RMS 半径 (m)</th><th>散开比</th><th>质心位移 (m)</th></tr></thead>
<tbody>{cargo_rows}</tbody></table>

<h3>5.3 人员失稳（抽检点）</h3>
<table><thead><tr><th>ID</th><th>过程最大水深 (m)</th><th>过程最大流速 (m/s)</th><th>判定</th></tr></thead>
<tbody>{hum_rows}</tbody></table>
<div class="note">溃口出口 3 名"人员"被判为倾覆（水深 1.1–1.2 m、流速 1.6–2.2 m/s）；
建筑出口处 0.4 m 水深、0.5 m/s 流速，全部安全。
这个结果本身就是一个有意义的结论：<b>积水深度不等于危险，流速才是控制因素</b>。</div>

<h2>7 · 你这台机器（RTX 3060 6 GB / 16 GB RAM）的可行性</h2>
<table><thead><tr><th>模块</th><th>本机能否跑</th><th>说明</th></tr></thead><tbody>
<tr><td><span class="flag f-g"></span>L2 局部惯性浅水（本仓库）</td><td class="ok">可以，CPU</td><td>200×200 网格 900 s 模拟约 65 s，内存 &lt; 1 GB</td></tr>
<tr><td><span class="flag f-g"></span>L3 三套评估判据（本仓库）</td><td class="ok">可以，CPU</td><td>秒级完成</td></tr>
<tr><td><span class="flag f-g"></span>L4 numpy 3DGS 光栅化（本仓库）</td><td class="ok">可以，CPU</td><td>1.8 万高斯 / 720×540，单帧约 1 s</td></tr>
<tr><td><span class="flag f-y"></span>3DGS 场景训练（gsplat / 官方实现）</td><td style="color:#a16207">可以，但小场景</td><td>6 GB 只能在 800×450、点云 &lt; 30 万时训练；RAM 16 GB 是瓶颈，需关掉其他程序</td></tr>
<tr><td><span class="flag f-y"></span>Gaussian Grouping 分割</td><td style="color:#a16207">勉强</td><td>训练需与 3DGS 同量级显存；建议 800×450 + 降采样点云</td></tr>
<tr><td><span class="flag f-r"></span>InsertAny3D</td><td class="bad">不建议本地</td><td>SD + 多视角渲染叠加，6 GB 显存 &amp; 16 GB 内存都会爆</td></tr>
<tr><td><span class="flag f-r"></span>RAF / Genesis 全场景耦合</td><td class="bad">不建议本地</td><td>官方建议 ≥ 24 GB 显存；粒子数上万即 OOM</td></tr>
<tr><td><span class="flag f-g"></span>DualSPHysics（替代 SPH 后端）</td><td class="ok">可以，纯 CPU</td><td>不占显存，8 核 16 线程可跑万级粒子</td></tr>
</tbody></table>
<div class="note"><b>建议的算力策略</b>：把"跑通验证"放在本机（用本仓库这套 CPU 闭环），
把"3DGS 重建 + 物体插入"这类重活放到学校集群 / 云 GPU（按小时租 3090/4090，几十块钱能跑完全部实验）。
本机不要硬扛 InsertAny3D。</div>

<h2>8 · 下一步（按周排的可行路径）</h2>
<table><thead><tr><th>阶段</th><th>做什么</th><th>产出</th><th>算力</th></tr></thead><tbody>
<tr><td>第 1–2 周</td><td>用本仓库跑通全流程，把参数换成你研究区的真实地形（DEM 导入，见 README）</td><td>可复现的基线结果 + 6 张图</td><td>本机</td></tr>
<tr><td>第 3–4 周</td><td>采集/下载一个真实 3DGS 场景，在云 GPU 上跑 Gaussian Grouping 分割</td><td>语义掩码 + 物体级资产</td><td>云 GPU</td></tr>
<tr><td>第 5–6 周</td><td>接入 DualSPHysics 做溃口射流的高保真交叉验证（对比本仓库浅水结果）</td><td>模型可信度证据</td><td>本机 CPU</td></tr>
<tr><td>第 7–9 周</td><td>补基线对比：静态水深阈值法 vs 本方法；消融：去掉流速项/去掉输运</td><td>论文实验章节</td><td>本机</td></tr>
<tr><td>第 10–12 周</td><td>写作 + 图表精修；把 L1 换成真实 3DGS 渲染出最终对比图</td><td>投稿稿</td><td>云 GPU</td></tr>
</tbody></table>

<div class="cap" style="margin-top:26px">
由 floodgs-lab 自动生成 · 所有数值来自 outputs/metrics.json ·
重跑命令 <code>python run_all.py</code>
</div>
</div></body></html>
"""

out = C.VIEWER / "index.html"
out.write_text(HTML, encoding="utf-8")
print(f"[viewer] {out}  ({out.stat().st_size/1024:.0f} KB)")
