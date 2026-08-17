#!/usr/bin/env python
"""統合28: locked透過段 (決定論的位相発火求心性 + 電気シナプスモデル) の
実回路張力基底を採取し、教師波形へLSQ再適合して飛行検証する。

透過段モデルの宣言 (ハリボテ回避のための正直な記載):
- 求心性の1周期1発・位相固定発火 = 桿状感覚子の実測生理 (Yarger & Fox 2018)
- pref反転2クラスタ = ストローク反転での発火集中 (同上)
- 電気シナプス8mV/0.5ms = ハルテア→b1電気結合 (Fayyazuddin & Dickinson 1996)。
  化学コネクトーム (MANC) に存在しないため実配線エッジの上にモデルとして追加。
配線 (どの求心性がどのMNへ至るか) は traced-connections の実データのまま。
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF

SETUP_KW = dict(afferent_mode="locked", pref_mode="reversal",
                electrical=True)
NB = 200


def measure_basis(T=0.8):
    """ω=0でlocked回路を走らせ、12筋の実張力波形(200bin)を採取"""
    net, mon, pg, pref, side, st_idx, n = PR.setup(**SETUP_KW)
    ch_idx = {mu: st_idx[mu] for mu in CF.NAMES if mu in st_idx}
    us = {mu: 0.0 for mu in ch_idx}
    fs = {mu: 0.0 for mu in ch_idx}
    acc = {mu: np.zeros(NB) for mu in ch_idx}
    cnt = np.zeros(NB)
    prev = 0
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        net.run(CF.DT_N * 1000 * _ms)
        nsp = mon.num_spikes
        if nsp > prev:
            for iN in np.array(mon.i[prev:nsp]):
                for mu, im in ch_idx.items():
                    if int(iN) == im:
                        us[mu] += 1.0 / CF.TAU_A
            prev = nsp
        b = min(int(((tn * PR.WBF) % 1.0) * NB), NB - 1)
        for mu in ch_idx:
            us[mu] -= CF.DT_N * us[mu] / CF.TAU_A
            fs[mu] += CF.DT_N * (us[mu] - fs[mu]) / CF.TAU_A
            if tn > 0.2:
                acc[mu][b] += fs[mu]
        if tn > 0.2:
            cnt[b] += 1
    return {mu: acc[mu] / np.maximum(cnt, 1) for mu in ch_idx}


def fit(Bc):
    """回路実基底で教師u1-4波形へLSQ適合"""
    M = CF.chan_matrix()
    uref = CF.UREF
    cols, names2, means = [], [], []
    for j, mu in enumerate(CF.NAMES):
        if mu not in Bc:
            continue
        dT = Bc[mu] - Bc[mu].mean()
        cols.append(np.outer(dT, M[j]).reshape(-1))
        names2.append(mu)
        means.append(Bc[mu].mean())
    A = np.stack(cols, axis=1)
    y = uref[:, 1:5].reshape(-1)
    g, *_r = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ g) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return g, names2, np.array(means), r2


def fly(gc, names2, means, mode="ff", T=2.0):
    """locked回路で飛行。mode='ff'はω=0、'closed'はハルテア位相シフト帰還"""
    env = PR._fly_env()
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    m = CF.OPT.get_model()
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
    M = CF.chan_matrix()
    jmap = [CF.NAMES.index(mu) for mu in names2]
    net, mon, pg, pref, side, st_idx, n = PR.setup(**SETUP_KW)
    ch_idx = {mu: st_idx[mu] for mu in names2 if mu in st_idx}
    us = {mu: 0.0 for mu in names2}
    fs = {mu: 0.0 for mu in names2}
    prev = 0
    shift_prev = np.zeros(len(pref))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    alive, ups = 0, []
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        om = d.qvel[3:6]
        if mode == "closed":
            o = np.clip(om, -12, 12)
            shift = PR.C_PHASE * (side * o[0] + o[1]
                                  + side * np.cos(2 * np.pi * pref) * o[2])
            pg.v = pg.v - (shift - shift_prev)
            shift_prev = shift
        net.run(CF.DT_N * 1000 * _ms)
        nsp = mon.num_spikes
        if nsp > prev:
            for iN in np.array(mon.i[prev:nsp]):
                for mu, im in ch_idx.items():
                    if int(iN) == im:
                        us[mu] += 1.0 / CF.TAU_A
            prev = nsp
        u14 = np.zeros(4)
        for jj, mu in enumerate(names2):
            us[mu] -= CF.DT_N * us[mu] / CF.TAU_A
            fs[mu] += CF.DT_N * (us[mu] - fs[mu]) / CF.TAU_A
            u14 += gc[jj] * (fs[mu] - means[jj]) * M[jmap[jj]]
        if tn < 0.2:   # 起動橋渡し (回路の張力立ち上がり待ち)
            phc = (tn * Pw["freq"]) % 1.0
            i = min(int(phc * NB), NB - 1)
            w = tn / 0.2
            u14 = (1 - w) * CF.Z["recon"][i] + w * u14
        u = np.array([CF.U0_CONST, *np.clip(u14, -0.55, 0.55)])
        for kp in range(int(CF.DT_N / dtp)):
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
                                                  + Pw["pitch_bias"]
                                                  + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = e * (-Pw["pitch_amp"] * rot
                                                   + Pw["pitch_bias"]
                                                   + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        alive = kn + 1
    return alive * CF.DT_N, float(np.mean(ups)) if ups else 0


if __name__ == "__main__":
    Bc = measure_basis()
    for mu, w in Bc.items():
        print(f"  {mu}: p-p={w.max()-w.min():.1f} mean={w.mean():.1f}",
              flush=True)
    gc, names2, means, r2 = fit(Bc)
    print(f"locked回路実基底LSQ: {len(names2)}筋 R²={r2:.3f}", flush=True)
    np.savez("outputs/locked_fit.npz", g=gc, names=np.array(names2),
             means=means, basis=np.stack([Bc[mu] for mu in names2]), r2=r2)
    s, up = fly(gc, names2, means, mode="ff")
    print(f"locked回路FF(ω=0)      生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    s, up = fly(gc, names2, means, mode="closed")
    print(f"locked回路閉ループ        生存{s:.2f}s 直立度{up:+.2f}", flush=True)
