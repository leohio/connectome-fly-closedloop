#!/usr/bin/env python
"""統合28 LoopC: コネクトーム高速ループ — 実VNC回路による翅波形の生成と変調。

実証済みの土台:
  - 教師ESの高速ループの正体 = ストローク位相同期の波形整形 (統合28 LoopA)
  - 実測MN位相+単収縮カーネルの筋合成FFが0.76s安定化 (LoopB, 基準0.24s)
本ループ: 波形をオフライン合成ではなく**実回路のスパイクから生成**する。
  ω → ハルテア位相符号 (C_PHASE=0.03, 飽和±12) → MANC 1,556細胞 0.1ms
  → 操舵MNスパイク → α単収縮 (τα=4.25ms, Azevedo) → 張力波形
  → 適合済みg_m (LoopB) で翅チャネルへ
条件: circuit(閉ループ) / circuit_ff(ω=0) / ideal_ff(基準) / shuffle
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, prefs

import phase_reflex as PR
import optics as OPT

prefs.codegen.target = "numpy"

DT_N = 0.0001
TAU_A = 0.00425

Z = np.load("outputs/muscle_ff.npz")
G = Z["g"]
NAMES = [str(n) for n in Z["names"]]
BASIS = Z["basis"]           # (n_mu, NB) 正規化張力波形
PHIS = Z["phis"]
UREF = Z["uref"]
U0_CONST = float(UREF[:, 0].mean())
BASIS_STD = BASIS.std(axis=1)          # 各筋の理想波形の変調幅


def chan_matrix():
    """筋→u1..u4 の解剖学写像 (LoopBと同一)"""
    chan_of = {"b1": "amp", "b2": "amp", "i1": "amp_neg", "i2": "amp_neg",
               "iii1": "aoa", "iii3": "aoa", "hg1": "ctr", "hg2": "ctr",
               "hg3": "ctr", "hg4": "ctr"}
    M = np.zeros((len(NAMES), 4))
    for k, mu in enumerate(NAMES):
        m_base, side = mu.rsplit("_", 1)
        ch = chan_of[m_base]
        sgn = -1.0 if ch == "amp_neg" else 1.0
        if ch in ("amp", "amp_neg"):
            M[k, 1] = sgn * (1 if side == "L" else -1)
        elif ch == "ctr":
            M[k, 0] = 0.5
        elif ch == "aoa":
            M[k, 2] = 0.5
            M[k, 3] = (1 if side == "L" else -1) * 0.5
    return M


def trial(mode="circuit", T=2.0, shuffle=None, seed=0):
    from brian2 import ms as _ms
    env = PR._fly_env()
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    m = OPT.get_model()
    aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                     "wing_yaw_right", "wing_roll_right",
                     "wing_pitch_right"]}
    M = chan_matrix()
    use_net = mode in ("circuit", "circuit_ff", "shuffle")
    if use_net:
        PR.C_PHASE = 0.03
        sh = "all" if mode == "shuffle" else None
        net, mon, pg, pref, side, st_idx, n = PR.setup(
            shuffle=sh, shuffle_seed=seed)
        ch_idx = {mu: st_idx[mu] for mu in NAMES if mu in st_idx}
        # 単収縮状態 (α関数 = 2段1次)
        us = {mu: 0.0 for mu in ch_idx}
        fs = {mu: 0.0 for mu in ch_idx}
        tbar = {mu: 0.0 for mu in ch_idx}
        tvar = {mu: 1e-6 for mu in ch_idx}
        prev_nsp = 0
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    NB = len(PHIS)
    alive = 0
    ups = []
    u14 = np.zeros(4)
    n_steps = int(T / DT_N)
    for kn in range(n_steps):
        tn = kn * DT_N
        om_true = d.qvel[3:6]
        if use_net:
            omg = (0.0, 0.0, 0.0) if mode == "circuit_ff" else \
                (om_true[0], om_true[1], om_true[2])
            pg.rates = PR.hal_rates(tn, omg, pref, side) * Hz
            net.run(DT_N * 1000 * _ms)
            nsp = mon.num_spikes
            if nsp > prev_nsp:
                ii = np.array(mon.i[prev_nsp:nsp])
                for iN in ii:
                    for mu, idx_mu in ch_idx.items():
                        if int(iN) == idx_mu:
                            us[mu] += 1.0 / TAU_A   # スパイク入力
                prev_nsp = nsp
            # α filter更新 + 基線/分散のEMA
            vec = np.zeros(len(NAMES))
            for j, mu in enumerate(NAMES):
                if mu not in ch_idx:
                    continue
                us[mu] -= DT_N * us[mu] / TAU_A
                fs[mu] += DT_N * (us[mu] - fs[mu]) / TAU_A
                tbar[mu] += DT_N * (fs[mu] - tbar[mu]) / 0.1
                dv = fs[mu] - tbar[mu]
                tvar[mu] += DT_N * (dv * dv - tvar[mu]) / 0.1
                scale = BASIS_STD[j] / max(np.sqrt(tvar[mu]), 1e-9)
                vec[j] = dv * scale
            u14 = G @ (vec[:, None] * M)      # (4,)
        else:
            phc = (tn * Pw["freq"]) % 1.0
            i = min(int(phc * NB), NB - 1)
            u14 = Z["recon"][i]
        # 起動0.2sは理想FFで橋渡し (回路の基線EMA安定待ち)
        if use_net and tn < 0.2:
            phc = (tn * Pw["freq"]) % 1.0
            i = min(int(phc * NB), NB - 1)
            w = tn / 0.2
            u14 = (1 - w) * Z["recon"][i] + w * u14
        u = np.array([U0_CONST, *np.clip(u14, -0.55, 0.55)])
        for kp in range(int(DT_N / dtp)):
            tt = tn + kp * dtp
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
    return alive * DT_N, float(np.mean(ups)) if ups else 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    conds = [("ideal_ff", "理想FF(基準0.76s)"),
             ("circuit_ff", "回路FF(ω=0)"),
             ("circuit", "回路閉ループ"),
             ("shuffle", "シャッフル対照")]
    if mode != "all":
        conds = [c for c in conds if c[0] == mode]
    for md, lbl in conds:
        s, up = trial(mode=md)
        print(f"{lbl:22s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
