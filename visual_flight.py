#!/usr/bin/env python
"""統合36 可視化: 飛行 + 腹髄発火 (実MANC座標) + 位相制御の3面統合動画。

  sim     : 全適応要素が生物学的学習の構成 (K=報酬学習, W=局所則) で飛行し、
            フレーム・全スパイク・筋位相フェーザ・復号ωを記録する
  compose : 1280x720 に合成 (左=飛行+位相パネル、右=腹髄点群のグロー発火)

腹髄パネルは MANC の実ソーマ座標 (1549/1556ニューロン) を使い、クラスで彩色。
これは飛行を実際に駆動している回路そのものの発火である (演出用の別データではない)。
"""
import sys
import json
import numpy as np

W_, H_ = 1280, 720
FW, FH = 640, 480
FPS = 60
T_FLY = 6.0
REC_DT = 0.001          # 記録間隔 [s]


def simulate():
    import mujoco
    import imageio
    from brian2 import ms as _ms, prefs
    prefs.codegen.target = "numpy"
    import phase_reflex as PR
    import connectome_fastloop as CF
    import locked_circuit as LC
    import openloop_hover as OH
    import connectome_bioflight as CB
    import plastic_readout as P
    import reward_K as RK

    res = json.load(open("outputs/reward_K.json"))
    sel = json.load(open("outputs/reward_K_selected.json"))["selected_seed"]
    th = np.array([r for r in res if r["seed"] == sel][0]["theta"])
    CB.K_POL = th[:CB.N_U * CB.N_X].reshape(CB.N_U, CB.N_X)
    CB.B_POL = th[CB.N_U * CB.N_X:]
    print("K=報酬学習 (個体%d), W=局所デルタ則を準備中..." % sel, flush=True)
    d0 = P.episodes(rng_seed=0, n_train=120)
    Wd, _ = P.learn(d0, rng_seed=0, checkpoints=[120])
    Sg, names, phi0 = Wd, d0["names"], d0["phi0"]

    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P0, U_TRIM = CB.P0, CB.U_TRIM
    net, mon, pg, pref, side, st_idx, n = PR.setup(**LC.SETUP_KW)
    idx_of = {j: st_idx[j] for j in names if j in st_idx}
    zc = {j: 0j for j in names}
    prev = 0
    shift_prev = np.zeros(len(pref))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    d.qvel[3:6] = np.random.default_rng(1).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, FH, FW)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 6.0, -8, 110
    dtp = m.opt.timestep
    kk = max(P0["sharp"], 1e-3)
    R = np.zeros(9)
    per = 1.0 / P0["freq"]
    NBOX = max(int(per / CF.DT_N), 1)
    ombuf = np.zeros((NBOX, 3))
    ombi, omsum = 0, np.zeros(3)
    u_cmd = np.array(U_TRIM, float)
    u = np.array(U_TRIM, float)
    om_est = np.zeros(3)
    t_next, next_f, next_r = 0.0, 0.0, 0.0
    frames = []
    rec = {"t": [], "om_est": [], "om_true": [], "phase": [],
           "zc_ang": [], "zc_mag": []}
    n_sub = max(int(round(CF.DT_N / dtp)), 1)
    TAU_TW = CB.TAU_TW
    for kn in range(int(T_FLY / CF.DT_N)):
        tn = kn * CF.DT_N
        omsum += d.qvel[3:6] - ombuf[ombi]
        ombuf[ombi] = d.qvel[3:6].copy()
        ombi = (ombi + 1) % NBOX
        o_slow = omsum / NBOX
        o = np.clip(o_slow, -12, 12)
        shift = PR.C_PHASE * (side * o[0] + o[1]
                              + side * np.cos(2 * np.pi * pref) * o[2])
        pg.v = pg.v - (shift - shift_prev)
        shift_prev = shift
        net.run(CF.DT_N * 1000 * _ms)
        ph_w = (tn * P0["freq"]) % 1.0
        nsp = mon.num_spikes
        if nsp > prev:
            ev = np.exp(2j * np.pi * ph_w)
            for iN in np.array(mon.i[prev:nsp]):
                for j, im in idx_of.items():
                    if int(iN) == im:
                        zc[j] += ev
            prev = nsp
        dphi = np.zeros(len(names))
        ok = True
        for q, j in enumerate(names):
            zc[j] -= CF.DT_N * zc[j] / LC.TAU_P
            if abs(zc[j]) < 1e-3:
                ok = False
                break
            dd = np.angle(zc[j]) / (2 * np.pi) - phi0[q]
            dphi[q] = (dd + 0.5) % 1.0 - 0.5
        if ok and tn > 0.12:
            om_est = dphi @ Sg
        if tn >= next_r:
            next_r += REC_DT
            rec["t"].append(tn)
            rec["om_est"].append(om_est.copy())
            rec["om_true"].append(o_slow.copy())
            rec["phase"].append(ph_w)
            rec["zc_ang"].append([float(np.angle(zc[j])) for j in names])
            rec["zc_mag"].append([float(abs(zc[j])) for j in names])
        if tn >= t_next:
            t_next += per
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            v = d.qvel[:3]
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          om_est[0] / 20.0, om_est[1] / 20.0,
                          om_est[2] / 20.0])
            u_cmd = np.clip(U_TRIM + np.tanh(CB.K_POL @ x + CB.B_POL) * 0.35,
                            -0.55, 0.55)
        for _ in range(n_sub):
            t = d.time
            u += dtp * (u_cmd - u) / TAU_TW
            amp = np.clip(1.0 + u[0], 0.5, 1.6)
            env0 = min(t / 0.03, 1.0)
            ph2 = 2 * np.pi * P0["freq"] * t
            s = np.sin(ph2)
            rot = np.tanh(kk * np.cos(ph2 + P0["phase"])) / np.tanh(kk)
            e = env0 * amp
            d.ctrl[:] = 0
            d.ctrl[aid["wing_yaw_left"]] = e * (P0["yaw_amp"] * s + u[1] + u[2])
            d.ctrl[aid["wing_yaw_right"]] = e * (P0["yaw_amp"] * s + u[1] - u[2])
            d.ctrl[aid["wing_pitch_left"]] = e * (-P0["pitch_amp"] * rot
                                                  + P0["pitch_bias"] + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = e * (-P0["pitch_amp"] * rot
                                                   + P0["pitch_bias"] + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = e * P0["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = e * P0["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if d.time >= next_f:
            next_f += 1.0 / FPS
            cam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            print(f"墜落 t={d.time:.2f}s", flush=True)
            break
        if kn % 5000 == 0:
            print(f"  t={tn:.2f}s z={d.qpos[2]:.1f}", flush=True)
    import imageio
    imageio.mimsave("outputs/vf_flight_raw.mp4", frames, fps=FPS, quality=8)
    sp_t = np.array(mon.t / _ms) / 1000.0
    sp_i = np.array(mon.i)
    np.savez("outputs/vf_record.npz",
             sp_t=sp_t, sp_i=sp_i,
             t=np.array(rec["t"]), om_est=np.array(rec["om_est"]),
             om_true=np.array(rec["om_true"]), phase=np.array(rec["phase"]),
             zc_ang=np.array(rec["zc_ang"]), zc_mag=np.array(rec["zc_mag"]),
             names=np.array(names), phi0=np.array(phi0),
             dur=d.time)
    print(f"記録完了: {len(frames)}フレーム, {len(sp_t)}スパイク, "
          f"生存{d.time:.2f}s", flush=True)


def neuron_layout():
    """実MANC座標を2Dへ射影する。回路1,556個に加え、腹髄の解剖学的な形を
    出すためMANC全ニューロン (約23k、実座標) を背景シルエットとして返す"""
    import pandas as pd
    import vnc_model
    ids_all, pre, post, w = vnc_model.build_arrays()
    sub = np.load("outputs/subcircuit_ids.npy")
    subset = set(int(s) for s in sub)
    keep = np.array([i for i, b in enumerate(ids_all) if int(b) in subset])
    ids = ids_all[keep]
    props = pd.read_feather("neuron-properties.feather")
    allpos = np.array([np.asarray(p, float) if p is not None
                       and not np.any(pd.isna(p)) else [np.nan]*3
                       for p in props["position"]])
    goodA = ~np.isnan(allpos[:, 0])
    PA = allpos[goodA]
    C0 = PA.mean(0)
    Cc = PA - C0
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    bg = Cc @ Vt[:2].T                       # 全ニューロン背景 (前後, 左右)
    pr = props.set_index("bodyId")
    pos = np.full((len(ids), 3), np.nan)
    cls = []
    for i, b in enumerate(ids):
        try:
            r = pr.loc[int(b)]
        except KeyError:
            cls.append("intrinsic neuron")
            continue
        p = r["position"]
        if p is not None and not np.any(pd.isna(p)):
            pos[i] = np.asarray(p, float)
        cls.append(str(r["class"]))
    good = ~np.isnan(pos[:, 0])
    xy2 = np.full((len(ids), 2), np.nan)
    xy2[good] = (pos[good] - C0) @ Vt[:2].T
    palette = {"sensory neuron": (56, 227, 238),
               "sensory ascending": (34, 197, 194),
               "motor neuron": (255, 149, 64),
               "intrinsic neuron": (150, 110, 245),
               "ascending neuron": (94, 234, 141),
               "descending neuron": (244, 114, 182),
               "efferent ascending": (250, 204, 21)}
    cols = [palette.get(c, (148, 163, 184)) for c in cls]
    return xy2, cols, ids, bg


def compose():
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    z = np.load("outputs/vf_record.npz", allow_pickle=True)
    sp_t, sp_i = z["sp_t"], z["sp_i"]
    rt, om_e, om_t = z["t"], z["om_est"], z["om_true"]
    zang, zmag = z["zc_ang"], z["zc_mag"]
    names = [str(x) for x in z["names"]]
    phi0 = z["phi0"]
    xy, cols, ids, bg = neuron_layout()
    ok = ~np.isnan(xy[:, 0])
    # 腹髄パネル座標 (右半分、長軸を縦に)。スケールは全ニューロン背景で決める
    a0, a1 = bg[:, 0].min(), bg[:, 0].max()
    b0, b1 = bg[:, 1].min(), bg[:, 1].max()
    PX0, PY0, PW, PH = 655, 40, 610, 660
    def to_px(u, v):
        return (PX0 + 110 + (v - b0) / (b1 - b0) * (PW - 220),
                PY0 + 20 + (u - a0) / (a1 - a0) * (PH - 40))
    bgx, bgy = to_px(bg[:, 0], bg[:, 1])
    px = np.full(len(xy), -1.0)
    py = np.full(len(xy), -1.0)
    px[ok], py[ok] = to_px(xy[ok, 0], xy[ok, 1])
    # スパイクをニューロンごとに時刻ソート
    order = np.argsort(sp_t)
    sp_t, sp_i = sp_t[order], sp_i[order]
    try:
        f_jp = ImageFont.truetype(
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 17)
        f_sm = ImageFont.truetype(
            "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", 13)
    except Exception:
        f_jp = f_sm = ImageFont.load_default()
    rd = imageio.get_reader("outputs/vf_flight_raw.mp4")
    wr = imageio.get_writer("outputs/vf_composite.mp4", fps=FPS, quality=8)
    BG = (9, 12, 32)
    MCOL = [(255, 99, 132), (255, 159, 64), (255, 205, 86), (75, 222, 192),
            (54, 162, 235), (153, 102, 255), (201, 240, 100), (255, 120, 200),
            (120, 220, 255)]
    n_fr = 0
    GLOW = 0.05                                   # 発火グローの残光 [s]
    j0 = 0
    for fi, fr in enumerate(rd):
        t = fi / FPS
        img = Image.new("RGB", (W_, H_), BG)
        dr = ImageDraw.Draw(img, "RGBA")
        img.paste(Image.fromarray(fr), (0, 60))
        dr.text((14, 8), "実MANCコネクトーム駆動飛行 (K=報酬学習, W=局所デルタ則)",
                font=f_jp, fill=(235, 238, 250))
        dr.text((14, 34), f"t = {t:4.2f}s", font=f_sm, fill=(160, 170, 200))
        # --- 腹髄点群 (背景=全MANC、前景=飛行回路) ---
        dr.text((PX0 + 30, 8), "MANC腹髄 — 飛行回路1,556 (実ソーマ座標)",
                font=f_jp, fill=(235, 238, 250))
        for i in range(0, len(bgx), 2):
            dr.point((bgx[i], bgy[i]), fill=(52, 60, 100, 255))
        for i in range(len(px)):
            if px[i] < 0:
                continue
            c = cols[i]
            dr.ellipse([px[i]-1.5, py[i]-1.5, px[i]+1.5, py[i]+1.5],
                       fill=(int(c[0]*0.55), int(c[1]*0.55), int(c[2]*0.55), 220))
        # --- 発火グロー ---
        while j0 < len(sp_t) and sp_t[j0] < t - GLOW:
            j0 += 1
        j1 = j0
        while j1 < len(sp_t) and sp_t[j1] <= t:
            j1 += 1
        for k in range(j0, j1):
            i = int(sp_i[k])
            if i >= len(px) or px[i] < 0:
                continue
            age = (t - sp_t[k]) / GLOW
            g = max(0.0, 1.0 - age)
            c = cols[i]
            r2 = 2.0 + 3.5 * g
            al = int(70 + 150 * g)
            dr.ellipse([px[i]-r2, py[i]-r2, px[i]+r2, py[i]+r2],
                       fill=(c[0], c[1], c[2], al))
            if g > 0.55:
                dr.ellipse([px[i]-1.4, py[i]-1.4, px[i]+1.4, py[i]+1.4],
                           fill=(255, 255, 255, 230))
        leg = [("感覚 (ハルテア他)", (56, 227, 238)),
               ("介在", (150, 110, 245)), ("運動 (操舵筋MN)", (255, 149, 64)),
               ("上行/下行", (94, 234, 141))]
        lx = PX0 + 12
        for txt, c in leg:
            dr.ellipse([lx, H_-26, lx+9, H_-17], fill=c)
            dr.text((lx + 14, H_-30), txt, font=f_sm, fill=(200, 208, 228))
            lx += 14 + 11 * len(txt) + 18
        # --- 位相制御パネル (左下) ---
        ri = min(int(t / REC_DT), len(rt) - 1)
        cx, cy, rad = 92, 628, 66
        dr.text((20, 548), "位相制御", font=f_jp, fill=(235, 238, 250))
        dr.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], outline=(90, 100, 140, 255),
                   width=2)
        ph = float(z["phase"][ri]) if ri < len(z["phase"]) else 0.0
        ang0 = 2 * np.pi * ph - np.pi / 2
        dr.ellipse([cx + (rad-6)*np.cos(ang0) - 5, cy + (rad-6)*np.sin(ang0) - 5,
                    cx + (rad-6)*np.cos(ang0) + 5, cy + (rad-6)*np.sin(ang0) + 5],
                   fill=(255, 255, 255, 240))
        for q in range(len(names)):
            base = 2 * np.pi * float(phi0[q]) - np.pi / 2
            dr.line([cx + (rad-2)*np.cos(base), cy + (rad-2)*np.sin(base),
                     cx + (rad+4)*np.cos(base), cy + (rad+4)*np.sin(base)],
                    fill=(120, 130, 160, 180), width=1)
            ang = float(zang[ri, q]) - np.pi / 2
            L = (0.25 + 0.65 * min(float(zmag[ri, q]), 1.0)) * (rad - 10)
            c = MCOL[q % len(MCOL)]
            dr.line([cx, cy, cx + L*np.cos(ang), cy + L*np.sin(ang)],
                    fill=(c[0], c[1], c[2], 235), width=3)
        dr.text((cx - 58, cy + rad + 6), "操舵筋の活動位相 (羽ばたき周期内)",
                font=f_sm, fill=(170, 178, 205))
        # --- ω トレース ---
        tx0, tx1 = 235, 630
        lab = ["roll", "pitch", "yaw"]
        for axq in range(3):
            ty = 560 + axq * 52
            dr.line([tx0, ty + 22, tx1, ty + 22], fill=(60, 68, 100, 200))
            dr.text((tx0 - 38, ty + 12), lab[axq], font=f_sm,
                    fill=(170, 178, 205))
            w0 = max(0.0, t - 1.2)
            m = (rt >= w0) & (rt <= t)
            if m.sum() > 2:
                ts = (rt[m] - w0) / 1.2
                for arr, cc, wd in [(om_t[m, axq], (235, 238, 250, 255), 2),
                                    (om_e[m, axq], (56, 227, 238, 255), 1)]:
                    v = np.clip(arr / 25.0, -1, 1)
                    pts = list(zip(tx0 + ts * (tx1 - tx0), ty + 22 - v * 20))
                    dr.line(pts, fill=cc, width=wd)
        dr.text((tx0 + 90, 545), "体角速度ω: 白=真値(遅い成分)  青=回路復号",
                font=f_sm, fill=(170, 178, 205))
        wr.append_data(np.asarray(img))
        n_fr += 1
    wr.close()
    print(f"合成完了 outputs/vf_composite.mp4 ({n_fr}フレーム)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "sim":
        simulate()
    else:
        compose()
