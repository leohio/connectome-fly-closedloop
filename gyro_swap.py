#!/usr/bin/env python
"""統合29 LoopI: 決定的実験 — 教師方策のジャイロ入力を回路復号ωに差し替える。

同一の教師方策 u = tanh(K x + b)·US を使い、x のω成分だけを
  (a) 真のジャイロ (基準3.0s)
  (b) 実MANC回路が復号したω (実ハルテア求心性→実配線→操舵MN位相→較正復号)
  (c) ω=0 (ジャイロ喪失)
  (d) シャッフル配線で復号したω (行動レベルの配線特異性の検定)
に差し替えて比較する。感覚経路だけを入れ替える対照実験。
"""
import sys
import numpy as np
import mujoco
from brian2 import ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC
from decode_fly import make_decode

N_X, N_U = 7, 5


def teacher_K():
    ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    th = ckp["theta"]
    return th[:N_U * N_X].reshape(N_U, N_X).copy(), th[N_U * N_X:]


def trial(src="true", T=3.0, pert_seed=None, shuffle=None, seed=0):
    env = PR._fly_env()
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    US = env["U_SCALE"]
    K, b = teacher_K()
    m = CF.OPT.get_model()
    aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                     "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]}
    use_net = src in ("circuit", "shuffle")
    # 真のωから羽ばたき振動だけを除いた条件 (完全精度・遅延ほぼ無し)。
    # 教師の3.0sが「体の回転を知っていること」に依るのか、
    # 「自分の羽ばたき反動振動を読めること」に依るのかを分ける対照
    NB2 = max(int((1.0 / Pw["freq"]) / 0.0001), 1)
    sbuf = np.zeros((NB2, 3))
    sbi = 0
    ssum = np.zeros(3)
    if use_net:
        Sg, phg, jg, _Bp = make_decode()
        kw = dict(LC.SETUP_KW)
        if src == "shuffle":
            kw["shuffle"] = "all"
            kw["shuffle_seed"] = seed
        net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
        idx_of = {j: st_idx[j] for j in jg if j in st_idx}
        zc = {j: 0j for j in jg}
        prev = 0
        shift_prev = np.zeros(len(pref))
        NBOX = max(int((1.0 / Pw["freq"]) / CF.DT_N), 1)
        ombuf = np.zeros((NBOX, 3))
        ombi = 0
        omsum = np.zeros(3)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    alive, ups = 0, []
    om_est = np.zeros(3)
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        om = d.qvel[3:6]
        if use_net:
            omsum += om - ombuf[ombi]
            ombuf[ombi] = om
            ombi = (ombi + 1) % NBOX
            o = np.clip(omsum / NBOX, -12, 12)
            shift = PR.C_PHASE * (side * o[0] + o[1]
                                  + side * np.cos(2 * np.pi * pref) * o[2])
            pg.v = pg.v - (shift - shift_prev)
            shift_prev = shift
            net.run(CF.DT_N * 1000 * _ms)
            ph_w = (tn * Pw["freq"]) % 1.0
            nsp = mon.num_spikes
            if nsp > prev:
                ev = np.exp(2j * np.pi * ph_w)
                for iN in np.array(mon.i[prev:nsp]):
                    for j, im in idx_of.items():
                        if int(iN) == im:
                            zc[j] += ev
                prev = nsp
            dphi = np.zeros(len(jg))
            ok = True
            for q, j in enumerate(jg):
                zc[j] -= CF.DT_N * zc[j] / LC.TAU_P
                if abs(zc[j]) < 1e-3:
                    ok = False
                    break
                dd = np.angle(zc[j]) / (2 * np.pi) - phg[q]
                dphi[q] = (dd + 0.5) % 1.0 - 0.5
            if ok and tn > 0.12:
                om_est = dphi @ Sg
        for kp in range(int(CF.DT_N / dtp)):
            tt = tn + kp * dtp
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc_v = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zc_v, ZT_W)
            v = d.qvel[:3]
            if src == "true":
                ow = d.qvel[3:6]
            elif src == "true_vib":
                # 羽ばたき振動成分のみ (体の回転を除去) — これだけで飛べるなら
                # 教師は「自分の振動」を制御信号にしていたと確定する
                ssum += d.qvel[3:6] - sbuf[sbi]
                sbuf[sbi] = d.qvel[3:6].copy()
                sbi = (sbi + 1) % NB2
                ow = d.qvel[3:6] - ssum / NB2
            elif src == "true_slow":
                ssum += d.qvel[3:6] - sbuf[sbi]
                sbuf[sbi] = d.qvel[3:6].copy()
                sbi = (sbi + 1) % NB2
                ow = ssum / NB2
            elif src == "zero":
                ow = np.zeros(3)
            else:
                ow = om_est
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u = np.tanh(K @ x + b) * US
            amp = np.clip(1.0 + u[0], 0.5, 1.6)
            env0 = min(tt / 0.03, 1.0)
            ph2 = 2 * np.pi * Pw["freq"] * tt
            s = np.sin(ph2)
            rot = np.tanh(kk * np.cos(ph2 + Pw["phase"])) / np.tanh(kk)
            e = env0 * amp
            d.ctrl[:] = 0
            d.ctrl[aid["wing_yaw_left"]] = e * (Pw["yaw_amp"] * s + u[1] + u[2])
            d.ctrl[aid["wing_yaw_right"]] = e * (Pw["yaw_amp"] * s + u[1] - u[2])
            d.ctrl[aid["wing_pitch_left"]] = e * (-Pw["pitch_amp"] * rot
                                                  + Pw["pitch_bias"] + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = e * (-Pw["pitch_amp"] * rot
                                                   + Pw["pitch_bias"] + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        alive = kn + 1
    return alive * CF.DT_N, float(np.mean(ups)) if ups else 0.0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "true"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    s, up = trial(src=which, pert_seed=None if which == "true" else None,
                  seed=seed)
    print(f"{which}(seed{seed}): 生存{s:.2f}s 直立{up:+.2f}", flush=True)
