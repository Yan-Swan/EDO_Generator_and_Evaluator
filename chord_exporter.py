#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
和弦导出器：输入 k-平均律 (k-TET) 与以数字表示的多个音（0 为根音），
例如十二平均律大三和弦 = [0, 4, 7]。

功能：
  1) 用与网页相同的“钢琴音色”合成音频，保存为 kTET_[a, b, c].wav
     音频结构：从前到后连续弹分解和弦（每个单音 0.5s，音间无间隔）；
     分解和弦结束后停顿 0.5s，再弹 1s 和弦。
     总时长 = len(notes)*0.5 + 0.5 + 1.0 秒。
  2) 生成一张透明底图片：圆周上等距排布 k 个点，0 在正上方、顺时针编号，
     和弦相邻项连线、末项连回 0，保存为 kTET_[a, b, c].png

用法：
  python chord_exporter.py --k 12 --notes 0,4,7
  python chord_exporter.py --k 31 --notes 0,10,18,25 --root 264.0 --outdir output
"""

import argparse
import os
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ===== 钢琴音色参数（与网页 playNote / offlinePlayNote 完全一致）=====
PARTIALS = [  # (泛音倍数 n, 幅度 a, 衰减系数 d)
    (1, 1.00, 1.00),
    (2, 0.55, 0.70),
    (3, 0.38, 0.55),
    (4, 0.22, 0.42),
    (5, 0.14, 0.32),
    (6, 0.09, 0.25),
]
PEAK = 0.16
SR = 44100          # 采样率
DUR = 1.0           # 和弦时长（秒）


def envelope_vec(t, a, d, dur):
    """复刻网页的指数包络：起音 4ms → 保持 → 指数衰减到 0.0001。"""
    hold = max(0.01, dur - 0.3)        # 对应网页的 0.3*beatSec（beatSec=1）
    decay_end = hold + 0.3 * d
    out = np.zeros_like(t)
    m1 = (t >= 0) & (t < 0.004)
    out[m1] = 0.0001 * (PEAK * a / 0.0001) ** (t[m1] / 0.004)
    m2 = (t >= 0.004) & (t < hold)
    out[m2] = PEAK * a
    m3 = (t >= hold) & (t < decay_end)
    out[m3] = PEAK * a * (0.0001 / (PEAK * a)) ** ((t[m3] - hold) / (decay_end - hold))
    return out


def pan_gains(pan):
    """等功率立体声声像（pan∈[-1,1]）。"""
    pan = max(-1.0, min(1.0, pan))
    lg = np.sqrt((1.0 - pan) / 2.0)
    rg = np.sqrt((1.0 + pan) / 2.0)
    return lg, rg


def synth_note(p, k, root, dur, pan=0.0):
    """合成单个音（钢琴音色），返回 (left, right, 采样数)。"""
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    sig_l = np.zeros(n)
    sig_r = np.zeros(n)
    f = root * (2.0 ** (p / k))              # 当前 K-EDO 下的频率
    lg, rg = pan_gains(pan)
    for (harm, a, d) in PARTIALS:
        env = envelope_vec(t, a, d, dur)
        wave = env * np.sin(2.0 * np.pi * f * harm * t)
        sig_l += wave * lg
        sig_r += wave * rg
    return sig_l, sig_r, n


def synth_chord(k, notes, dur, root=264.0):
    """合成和弦（多音同时），返回 (left, right) 两个 numpy 数组。"""
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    sig_l = np.zeros(n)
    sig_r = np.zeros(n)
    for ni, p in enumerate(notes):
        f = root * (2.0 ** (p / k))              # 当前 K-EDO 下的频率
        pan = -0.5 if ni % 2 == 0 else 0.5       # 与网页旋律一致地交替声像
        lg, rg = pan_gains(pan)
        for (harm, a, d) in PARTIALS:
            env = envelope_vec(t, a, d, dur)
            wave = env * np.sin(2.0 * np.pi * f * harm * t)
            sig_l += wave * lg
            sig_r += wave * rg
    return sig_l, sig_r


def synth_sequence(k, notes, root=264.0):
    """按新结构合成整段音频（返回 left, right）：
       从前到后：每个单音连续各弹 0.5s（分解和弦，音与音之间无间隔）；
       分解和弦结束后停顿 0.5s，再弹 1s 和弦。
    """
    single = 0.5           # 每个单音时长
    gap = 0.5              # 分解和弦与末尾和弦之间的停顿
    chord_dur = 1.0        # 末尾和弦时长
    m = len(notes)
    total = m * single + gap + chord_dur
    N = int(round(total * SR))
    out_l = np.zeros(N)
    out_r = np.zeros(N)

    # 分解和弦：单音连续排列（无间隔）
    for i, p in enumerate(notes):
        sl, sr, n = synth_note(p, k, root, single, pan=0.0)
        start = int(round(i * single * SR))
        out_l[start:start + n] += sl
        out_r[start:start + n] += sr

    # 停顿 gap 之后，弹 1 秒和弦
    cl, cr = synth_chord(k, notes, chord_dur, root)
    cstart = int(round((m * single + gap) * SR))
    out_l[cstart:cstart + len(cl)] += cl
    out_r[cstart:cstart + len(cr)] += cr

    # 归一化，避免多音叠加削波
    peak = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-9)
    if peak > 0.9:
        out_l *= 0.9 / peak
        out_r *= 0.9 / peak
    return out_l, out_r


def write_wav(path, sig_l, sig_r, sr=SR):
    int_l = np.int16(np.clip(sig_l, -1.0, 1.0) * 32767)
    int_r = np.int16(np.clip(sig_r, -1.0, 1.0) * 32767)
    frames = np.column_stack((int_l, int_r)).tobytes()
    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)


def draw_diagram(k, notes, path):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    fig.patch.set_alpha(0.0)              # 透明底
    ax.set_aspect("equal")
    ax.axis("off")

    R = 1.0
    idxs = np.arange(k)
    # 0 在正上方，顺时针：角度 = π/2 - 2π*i/k（数学坐标，y 向上）
    angles = np.pi / 2 - 2.0 * np.pi * idxs / k
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)

    # 外圈
    ax.add_patch(Circle((0, 0), R, fill=False, edgecolor="#888888", lw=1.2, zorder=1))

    # 全部 k 个点与编号（顺时针）
    ax.scatter(xs, ys, s=28, c="#888888", zorder=3)
    fs = max(5, int(220 / k))
    for i in range(k):
        ax.annotate(
            str(i),
            (xs[i], ys[i]),
            xytext=(1.16 * np.cos(angles[i]), 1.16 * np.sin(angles[i])),
            ha="center", va="center", fontsize=fs, color="#666666",
        )

    # 和弦多边形：相邻项连线，末项连回 0
    seq = [p % k for p in notes] + [0]
    px = [xs[s] for s in seq]
    py = [ys[s] for s in seq]
    ax.plot(px, py, "-", color="#111111", lw=2.4, zorder=2)

    # 高亮和弦音
    cx = [xs[p % k] for p in notes]
    cy = [ys[p % k] for p in notes]
    ax.scatter(cx, cy, s=80, c="#111111", zorder=4)
    # 高亮根音 0（正上方）
    ax.scatter([xs[0]], [ys[0]], s=110, c="#111111", zorder=5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="k-TET 和弦 WAV / 透明图导出器")
    ap.add_argument("--k", type=int, required=True, help="平均律划分数量，如 12")
    ap.add_argument("--notes", type=str, required=True,
                    help="以逗号分隔的音高数字（0 为根音），如 0,4,7")
    ap.add_argument("--root", type=float, default=264.0,
                    help="根音频率(Hz)，默认 264.0 (A3)")
    ap.add_argument("--outdir", type=str, default="chord_output",
                    help="输出目录，默认 ./chord_output")
    args = ap.parse_args()

    k = args.k
    if k < 1:
        raise SystemExit("k 必须 >= 1")
    notes = [int(x) for x in args.notes.split(",") if x.strip() != ""]
    if not notes:
        raise SystemExit("notes 不能为空")

    os.makedirs(args.outdir, exist_ok=True)
    base = f"{k}TET_{notes}"          # 例如 12TET_[0, 4, 7]
    wav_path = os.path.join(args.outdir, base + ".wav")
    png_path = os.path.join(args.outdir, base + ".png")

    sig_l, sig_r = synth_sequence(k, notes, root=args.root)
    write_wav(wav_path, sig_l, sig_r)
    draw_diagram(k, notes, png_path)

    print(f"已生成：\n  {wav_path}\n  {png_path}")


if __name__ == "__main__":
    main()
