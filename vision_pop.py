#!/usr/bin/env python
"""統合23: 較正型の集団デコード — 読み出しSNRの壁への挑戦。

統合22の結論: 視覚→DNの姿勢符号化は実在するが、2〜6本のDN読み出しでは
ポアソンノイズが信号を圧倒する (SNR∝√N)。本統合では
  読み出し: サブ回路の非感覚ニューロン全部 (OCG+中継+DN ≈166本)
  デコーダ: 飛行前較正 — 既知の傾き(roll±CAL, pitch±CAL)を提示し
            各ニューロンのチューニングベクトル t を測定、整合フィルタ
            ŝ = Σ t_i(r_i−r0_i) / Σ t_i² で姿勢を推定 (単位: rad)
手続きはREAL/SHUFに同一適用 = 「回路が読み出し層まで姿勢情報を運ぶか」の
情報量ベースの公平な検定。運べない回路では t≈ノイズ → 反射は自然に消灯。

mode=quality : デコード品質 (較正→未知傾きの推定誤差) をREAL/SHUFで比較
mode=smoke   : RATE基盤上の飛行スモーク
mode=main    : n=8 本実験 (RATE / +VIS-POP-REAL / +VIS-POP-SHUF s0,s1)
"""
import sys
import numpy as np
import mujoco
from brian2 import SpikeMonitor, Network, Hz, ms, prefs

import phase_reflex as PR
import brain_visual as BV

prefs.codegen.target = "numpy"

DT_W = 0.002
TAU_DN = 0.1          # 集団なので遅く滑らかに
CAL_TILT = 0.2        # 較正提示の傾き [rad]
CAL_T = 0.4           # 較正1条件の時間 [s]
T_CAP = 3.0


def build(shuffle=None, seed=0):
    neu, syn, ids, idx, meta = BV.build_subcircuit(shuffle=shuffle,
                                                   shuffle_seed=seed)
    pg, sh, ssign, is_oc = BV.sensory_attach(neu, idx, meta)
    mon = SpikeMonitor(neu, record=False)
    net = Network(neu, syn, mon, pg, sh)
    sens = set(int(b) for b in np.concatenate([meta["ocr"], meta["vshs"]]))
    readout = np.array([i for b, i in idx.items() if b not in sens])
    return net, mon, pg, ssign, is_oc, readout


def calibrate(net, mon, pg, ssign, is_oc, readout):
    """5条件提示 → (r0, t_roll, t_pitch, s2_roll, s2_pitch)"""
    from brian2 import ms as _ms
    conds = [(0.0, 0.0), (CAL_TILT, 0.0), (-CAL_TILT, 0.0),
             (0.0, CAL_TILT), (0.0, -CAL_TILT)]
    rates = []
    prev = np.zeros(mon.source.N)
    for ro, pi in conds:
        pg.rates = BV.vis_rates(ro, pi, 0, 0, ssign, is_oc) * Hz
        net.run(CAL_T * 1000 * _ms)
        c = mon.count[:].copy()
        rates.append((c - prev)[readout] / CAL_T)
        prev = c
    r0, rp, rm, pp, pm = rates
    t_roll = (rp - rm) / (2 * CAL_TILT)     # Hz/rad
    t_pitch = (pp - pm) / (2 * CAL_TILT)
    s2r = float((t_roll ** 2).sum())
    s2p = float((t_pitch ** 2).sum())
    return r0, t_roll, t_pitch, max(s2r, 1e-9), max(s2p, 1e-9)


def decode(dr_rates, r0, t, s2):
    return float(np.dot(t, dr_rates - r0) / s2)


def trial(G_roll, G_pitch, vision=True, shuffle=None, seed=0, T=T_CAP,
          transfer_w=None, oracle=False):
    """RATEダンピング基盤 + 集団デコード視覚反射。(生存s, 直立度)
    transfer_w=(t_r, t_p, s2r, s2p): 固定重み (REAL較正) を強制 —
    生物の「読み出しは解剖学的に固定」の対照。r0は自条件の水平較正値。"""
    from brian2 import ms as _ms
    env = PR._fly_env()
    b_pol, U_SCALE = env["b_pol"], env["U_SCALE"]
    Pw, Q0, ZT_W, m = env["Pw"], env["Q0"], env["ZT_W"], env["m"]
    ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    K = ckp["theta"][:5 * 7].reshape(5, 7).copy()
    K[:, 2:4] = 0.0                  # 姿勢項ゼロ = ωダンピング+高度のみ
    if oracle:
        vision = False
        f_est = np.array([0.0, 0.0])
    if vision:
        net, mon, pg, ssign, is_oc, readout = build(shuffle, seed)
        r0, t_r, t_p, s2r, s2p = calibrate(net, mon, pg, ssign, is_oc,
                                           readout)
        if transfer_w is not None:
            t_r, t_p, s2r, s2p = transfer_w
        f_est = np.array([0.0, 0.0])         # [roll̂, pitcĥ] 低域推定
        prev_c = mon.count[:].copy()
        f_rate = r0.copy()
    d = mujoco.MjData(m)
    dtp = m.opt.timestep
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    gdL = gdR = dr = 0.0
    ups, n_alive = [], 0
    kk = max(Pw["sharp"], 1e-3)
    for wi in range(int(T / DT_W)):
        t = wi * DT_W
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
        if oracle:   # 真の姿勢+同じτ/ゲイン/チャネル (神経推定の上限)
            f_est += DT_W * (np.array([e_b[0], e_b[1]]) - f_est) / TAU_DN
            gdL = float(np.clip(G_roll * f_est[0], -0.4, 0.4))
            gdR = -gdL
            dr = float(np.clip(G_pitch * f_est[1], -0.4, 0.4))
        if vision:
            om = d.qvel[3:6]
            pg.rates = BV.vis_rates(e_b[0], e_b[1], om[0], om[1],
                                    ssign, is_oc) * Hz
            net.run(DT_W * 1000 * _ms)
            c = mon.count[:].copy()
            r_now = (c - prev_c)[readout] / DT_W
            prev_c = c
            f_rate += DT_W * (r_now - f_rate) / TAU_DN
            est = np.array([decode(f_rate, r0, t_r, s2r),
                            decode(f_rate, r0, t_p, s2p)])
            f_est = est
            gdL = float(np.clip(G_roll * f_est[0], -0.4, 0.4))
            gdR = -gdL
            dr = float(np.clip(G_pitch * f_est[1], -0.4, 0.4))
        for kp in range(int(DT_W / dtp)):
            tt = t + kp * dtp
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            eb2 = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            omp = d.qvel[3:6]
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -d.qvel[2] / 30.0,
                          eb2[0], eb2[1],
                          omp[0] / 20.0, omp[1] / 20.0, omp[2] / 20.0])
            u = np.tanh(K @ x + b_pol) * U_SCALE
            amp = np.clip(1.0 + u[0], 0.5, 1.6)
            env0 = min(tt / 0.03, 1.0)
            ph2 = 2 * np.pi * Pw["freq"] * tt
            s = np.sin(ph2)
            rot = np.tanh(kk * np.cos(ph2 + Pw["phase"] + dr)) / np.tanh(kk)
            down = 1.0 if np.cos(ph2) < 0 else 0.0
            eL = env0 * amp * (1.0 + gdL * down)
            eR = env0 * amp * (1.0 + gdR * down)
            d.ctrl[:] = 0
            d.ctrl[aid["wing_yaw_left"]] = eL * (Pw["yaw_amp"] * s + u[1] + u[2])
            d.ctrl[aid["wing_yaw_right"]] = eR * (Pw["yaw_amp"] * s + u[1] - u[2])
            d.ctrl[aid["wing_pitch_left"]] = eL * (-Pw["pitch_amp"] * rot
                                                   + Pw["pitch_bias"] + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = eR * (-Pw["pitch_amp"] * rot
                                                    + Pw["pitch_bias"] + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = eL * Pw["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = eR * Pw["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        n_alive = wi + 1
    return n_alive * DT_W, float(np.mean(ups)) if ups else 0.0


def mode_quality():
    """デコード品質: 較正後、未知の傾きを提示して推定値を読む"""
    from brian2 import ms as _ms
    for name, sh, seed in [("REAL", None, 0), ("SHUF s0", "all", 0),
                           ("SHUF s1", "all", 1)]:
        net, mon, pg, ssign, is_oc, readout = build(sh, seed)
        r0, t_r, t_p, s2r, s2p = calibrate(net, mon, pg, ssign, is_oc,
                                           readout)
        print(f"{name}: 読み出し{len(readout)}本 "
              f"|t_roll|²={s2r:.0f} |t_pitch|²={s2p:.0f}", flush=True)
        prev = mon.count[:].copy()
        for ro, pi in [(0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1),
                       (0.15, -0.1)]:
            pg.rates = BV.vis_rates(ro, pi, 0, 0, ssign, is_oc) * Hz
            net.run(300 * _ms)
            c = mon.count[:].copy()
            r = (c - prev)[readout] / 0.3
            prev = c
            er = decode(r, r0, t_r, s2r)
            ep = decode(r, r0, t_p, s2p)
            print(f"  真値(roll={ro:+.2f}, pitch={pi:+.2f}) → "
                  f"推定({er:+.3f}, {ep:+.3f})", flush=True)


def mode_smoke():
    s, up = trial(0, 0, vision=False, T=1.5)
    print(f"RATEのみ: 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    for gr, gp in [(1.0, -1.0), (2.0, -2.0), (0.5, -0.5)]:
        s, up = trial(gr, gp, T=1.5)
        print(f"POP G=({gr},{gp}): 生存{s:.2f}s 直立度{up:+.2f}", flush=True)


def mode_main():
    NREP = 8
    Gr, Gp = (eval(sys.argv[2]) if len(sys.argv) > 2 else (1.0, -1.0))
    print(f"=== 統合23本実験 (n={NREP}, G=({Gr},{Gp})) ===", flush=True)
    s, up = trial(0, 0, vision=False)
    print(f"{'RATEのみ':14s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    rows = [("RATE", [s], [up])]
    # REAL較正の固定重み (TRANSFER対照用)
    net, mon, pg, ssign, is_oc, readout = build(None, 0)
    _, t_r, t_p, s2r, s2p = calibrate(net, mon, pg, ssign, is_oc, readout)
    real_w = (t_r, t_p, s2r, s2p)
    conds = [("POP-REAL", None, 0, None),
             ("SHUF-adapt s0", "all", 0, None),
             ("SHUF-adapt s1", "all", 1, None),
             ("SHUF-fix s0", "all", 0, real_w)]
    for name, sh, seed, tw in conds:
        ss, uu = [], []
        for rep in range(NREP):
            s, up = trial(Gr, Gp, shuffle=sh, seed=seed, transfer_w=tw)
            ss.append(s); uu.append(up)
            print(f"  {name} rep{rep}: 生存{s:.2f}s 直立度{up:+.2f}",
                  flush=True)
        print(f"{name:14s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"直立度{np.mean(uu):+.3f}±SEM{np.std(uu)/np.sqrt(NREP):.3f}",
              flush=True)
        rows.append((name, ss, uu))
    np.savez("outputs/vision_pop.npz", rows=np.array(rows, dtype=object))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "quality"
    {"quality": mode_quality, "smoke": mode_smoke, "main": mode_main}[mode]()
