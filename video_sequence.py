#!/usr/bin/env python
"""統合39 可視化: 脳を中心とした神経系の発火 + 身体行動の並列動画。

右パネル (神経系):
  上=FlyWire脳 (実座標139k点のシルエット + 視覚サブ回路 + 指令DN群)
  下=MANC腹髄 (実座標23k点のシルエット + 歩行net/飛行回路の発火)
左パネル: flybody の行動 (歩行→離陸→飛行→降下→接地→歩行) + フェーズ帯。
すべて同一シミュレーションの実スパイク。
"""
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

W_, H_ = 1280, 720
FPS = 60
GLOW = 0.05
PHASE_NAMES = {0: "歩行 (MDN)", 1: "離陸 (GF)", 2: "飛行 [空中開始]", 3: "降下", 4: "着陸・静定"}
PHASE_COL = {0: (94, 234, 141), 1: (250, 204, 21), 2: (56, 227, 238),
             3: (255, 159, 64), 4: (200, 160, 255)}


def brain_layout():
    import pandas as pd
    co = pd.read_csv("../fly-brain-sim/Drosophila_brain_model/coordinates.csv")
    co = co.drop_duplicates("root_id")
    P = np.array([[float(x) for x in str(p).strip("[]").split()]
                  for p in co.position])
    rid = co.root_id.values
    C = P - P.mean(0)
    U, S, Vt = np.linalg.svd(C[np.random.default_rng(0).choice(
        len(C), 20000, replace=False)], full_matrices=False)
    XY = C @ Vt[:2].T
    pos = {int(r): XY[i] for i, r in enumerate(rid)}
    return XY, pos


def vnc_layout():
    import pandas as pd
    props = pd.read_feather("neuron-properties.feather")
    allpos = np.array([np.asarray(p, float) if p is not None
                       and not np.any(pd.isna(p)) else [np.nan] * 3
                       for p in props["position"]])
    good = ~np.isnan(allpos[:, 0])
    PA = allpos[good]
    C0 = PA.mean(0)
    U, S, Vt = np.linalg.svd(PA - C0, full_matrices=False)
    XY = (PA - C0) @ Vt[:2].T
    pos = {int(b): XY[i] for i, b in enumerate(props.bodyId.values[good])}
    return XY, pos


def compose():
    z = np.load("outputs/seq_record.npz", allow_pickle=True)
    phase = z["phase"]
    # --- 脳 ---
    bXY, bpos = brain_layout()
    import brain_visual as BV
    neu, syn, vids, vidx, meta = BV.build_subcircuit()
    from behavior_sequence import CMD_IDS
    cmd_ids = [int(b) for b in z["cmd_ids"]]
    # --- 腹髄 ---
    nXY, npos = vnc_layout()
    walk_ids = [int(b) for b in z["walk_ids"]]
    import vnc_model
    ids_all, pre, post, w = vnc_model.build_arrays()
    sub = np.load("outputs/subcircuit_ids.npy")
    subset = set(int(s) for s in sub)
    keep = [i for i, b in enumerate(ids_all) if int(b) in subset]
    fly_ids = [int(ids_all[i]) for i in keep]
    # パネル座標系
    BX0, BY0, BW, BH = 645, 34, 630, 400
    NX0, NY0, NW, NH = 700, 452, 520, 258
    def to_b(xy):
        a0, a1 = bXY[:, 0].min(), bXY[:, 0].max()
        b0, b1 = bXY[:, 1].min(), bXY[:, 1].max()
        return (BX0 + (xy[..., 0] - a0) / (a1 - a0) * BW,
                BY0 + (xy[..., 1] - b0) / (b1 - b0) * BH)
    def to_n(xy):
        a0, a1 = nXY[:, 0].min(), nXY[:, 0].max()
        b0, b1 = nXY[:, 1].min(), nXY[:, 1].max()
        # 腹髄は横倒し (長軸=横)
        return (NX0 + (xy[..., 0] - a0) / (a1 - a0) * NW,
                NY0 + (xy[..., 1] - b0) / (b1 - b0) * NH)
    rng = np.random.default_rng(1)
    bsub = rng.choice(len(bXY), 15000, replace=False)
    bsx, bsy = to_b(bXY[bsub])
    nsub = rng.choice(len(nXY), 12000, replace=False)
    nsx, nsy = to_n(nXY[nsub])
    vis_px = {}
    for i, b in enumerate(vids):
        if int(b) in bpos:
            vis_px[i] = to_b(np.array(bpos[int(b)]))
    cmd_px = {}
    for i, b in enumerate(cmd_ids):
        if b in bpos:
            cmd_px[i] = to_b(np.array(bpos[b]))
    cmd_kind = {}
    at = 0
    for k, v in CMD_IDS.items():
        for q in range(len(v)):
            cmd_kind[at + q] = k
        at += len(v)
    walk_px = {i: to_n(np.array(npos[b])) for i, b in enumerate(walk_ids)
               if b in npos}
    fly_px = {i: to_n(np.array(npos[b])) for i, b in enumerate(fly_ids)
              if b in npos}
    CMD_COL = {"MDN": (244, 114, 182), "GF": (250, 204, 21),
               "DNp09": (94, 234, 141)}
    try:
        f_jp = ImageFont.truetype(
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 17)
        f_sm = ImageFont.truetype(
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 13)
    except Exception:
        f_jp = f_sm = ImageFont.load_default()
    # スパイクを時刻ソート
    def sorted_sp(tk, ik):
        o = np.argsort(z[tk])
        return z[tk][o], z[ik][o]
    sp = {k: sorted_sp(f"{k}_t", f"{k}_i")
          for k in ("cmd", "walk", "fly", "vis")}
    # 飛行netの時刻は飛行開始からの相対 → 絶対へ
    tf0 = float(z["t_flight0"])
    ptr = {k: 0 for k in sp}
    BG = (9, 12, 32)
    rd = imageio.get_reader("outputs/seq_body.mp4")
    wr = imageio.get_writer("outputs/seq_composite.mp4", fps=FPS, quality=8)
    T_end = phase[-1][0]
    for fi, fr in enumerate(rd):
        t = phase[min(fi, len(phase) - 1)][0]
        ph_id = int(phase[min(fi, len(phase) - 1)][1])
        img = Image.new("RGB", (W_, H_), BG)
        dr = ImageDraw.Draw(img, "RGBA")
        img.paste(Image.fromarray(fr), (0, 60))
        dr.text((14, 8), "統合個体: 歩行 → [カット] 安定飛行 → 降下 → 軟着陸 → 歩行"
                " (遷移=DN発火)", font=f_jp, fill=(235, 238, 250))
        if 2.1 < t < 3.6:
            dr.rectangle([0, 60, 640, 540], fill=(0, 0, 0, 90))
            dr.text((120, 280), "[カット] 地上離陸は未解決 → 空中 (z=12) から飛行開始",
                    font=f_jp, fill=(255, 230, 120))
        pc = PHASE_COL[ph_id]
        dr.rectangle([14, 34, 200, 54], fill=(pc[0], pc[1], pc[2], 60),
                     outline=pc)
        dr.text((22, 36), f"{PHASE_NAMES[ph_id]}  t={t:4.1f}s", font=f_sm,
                fill=(240, 244, 255))
        # フェーズ帯
        y0 = 560
        dr.text((14, y0 - 20), "行動タイムライン", font=f_sm,
                fill=(170, 178, 205))
        for i2 in range(len(phase) - 1):
            xx0 = 14 + phase[i2][0] / T_end * 600
            xx1 = 14 + phase[i2 + 1][0] / T_end * 600
            c = PHASE_COL[int(phase[i2][1])]
            dr.rectangle([xx0, y0, xx1 + 1, y0 + 16], fill=c)
        dr.polygon([(14 + t / T_end * 600 - 5, y0 + 26),
                    (14 + t / T_end * 600 + 5, y0 + 26),
                    (14 + t / T_end * 600, y0 + 18)], fill=(255, 255, 255))
        yl = y0 + 34
        for pid in (0, 1, 2, 3, 4):
            c = PHASE_COL[pid]
            dr.rectangle([14 + pid * 122, yl, 26 + pid * 122, yl + 12], fill=c)
            dr.text((30 + pid * 122, yl - 2), PHASE_NAMES[pid], font=f_sm,
                    fill=(200, 208, 228))
        # --- 脳パネル ---
        dr.text((BX0 + 40, 8), "FlyWire脳 (実座標) — 視覚回路と下行指令DN",
                font=f_jp, fill=(235, 238, 250))
        for q in range(len(bsx)):
            dr.point((bsx[q], bsy[q]), fill=(46, 54, 92, 255))
        for i2, (px, py) in vis_px.items():
            dr.ellipse([px - 1.3, py - 1.3, px + 1.3, py + 1.3],
                       fill=(35, 90, 110, 220))
        for i2, (px, py) in cmd_px.items():
            c = CMD_COL[cmd_kind[i2]]
            dr.ellipse([px - 3, py - 3, px + 3, py + 3],
                       fill=(c[0] // 2, c[1] // 2, c[2] // 2, 255),
                       outline=(c[0], c[1], c[2], 255))
        # --- 腹髄パネル ---
        dr.text((NX0 - 40, NY0 - 22), "MANC腹髄 (実座標) — 歩行回路と飛行回路",
                font=f_jp, fill=(235, 238, 250))
        for q in range(len(nsx)):
            dr.point((nsx[q], nsy[q]), fill=(40, 48, 84, 255))
        # --- 発火グロー ---
        def draw_spikes(kind, px_map, col=None, base_r=2.0, tshift=0.0,
                        kindmap=None):
            tt, ii = sp[kind]
            j = ptr[kind]
            while j < len(tt) and tt[j] + tshift < t - GLOW:
                j += 1
            ptr[kind] = j
            k2 = j
            while k2 < len(tt) and tt[k2] + tshift <= t:
                i3 = int(ii[k2])
                k2 += 1
                if i3 not in px_map:
                    continue
                px, py = px_map[i3]
                g = max(0.0, 1.0 - (t - (tt[k2 - 1] + tshift)) / GLOW)
                c = col or (CMD_COL[kindmap[i3]] if kindmap else (255, 255, 255))
                r2 = base_r + 3.0 * g
                al = int(80 + 160 * g)
                dr.ellipse([px - r2, py - r2, px + r2, py + r2],
                           fill=(c[0], c[1], c[2], al))
                if g > 0.6:
                    dr.ellipse([px - 1.2, py - 1.2, px + 1.2, py + 1.2],
                               fill=(255, 255, 255, 230))
        draw_spikes("cmd", cmd_px, base_r=4.0, kindmap=cmd_kind)
        draw_spikes("vis", vis_px, col=(56, 227, 238))
        draw_spikes("walk", walk_px, col=(150, 110, 245), base_r=1.6)
        draw_spikes("fly", fly_px, col=(255, 149, 64), base_r=1.6,
                    tshift=0.0)
        lx = BX0 + 20
        for txt, c in [("MDN(後退)", CMD_COL["MDN"]), ("GF(離陸)", CMD_COL["GF"]),
                       ("視覚回路", (56, 227, 238)), ("歩行回路", (150, 110, 245)),
                       ("飛行回路", (255, 149, 64))]:
            dr.ellipse([lx, H_ - 24, lx + 9, H_ - 15], fill=c)
            dr.text((lx + 13, H_ - 28), txt, font=f_sm, fill=(200, 208, 228))
            lx += 13 + 11 * len(txt) + 14
        wr.append_data(np.asarray(img))
    wr.close()
    print("合成完了 outputs/seq_composite.mp4", flush=True)


if __name__ == "__main__":
    compose()
