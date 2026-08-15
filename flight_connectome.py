#!/usr/bin/env python
"""(a) コネクトーム飛行: ハルテア→MANC→操舵筋→翅の実配線反射で飛ぶ。

ループ構成 (2ms窓):
  flybody の体角速度 ω → ハルテア求心性レート(ジャイロ変調, 羽ばたき位相固定)
  → MANC 23,188ニューロン LIF → 操舵筋MN 22本のスパイク
  → 筋活動 (α核, 実測タイプの単収縮) → Melis準拠の翅運動学写像
  → flybody 翅駆動 → 物理 → ω → ...

翅運動学写像 (Melis et al. 2024 の機能同定の線形蒸留):
  振幅(片側)   ∝ +(b1+b2) − (b3+i1)     [基礎骨片筋↑, 拮抗筋↓]
  ストローク中心 ∝ hg群 (第4腋骨片筋)
  迎角バイアス  ∝ iii群 (第3腋骨片筋)
  変調は各筋の基準発火率からの相対値 Δr/r0 (MANC左右完成度差の補正)

条件比較 (z=12から1.5s):
  A) 反射OFF (ハルテアのジャイロ変調なし)  = 開ループ
  B) 反射ON  (コネクトーム反射のみ、学習なし)
"""
import numpy as np
import pandas as pd
import mujoco
from brian2 import ms, Hz, prefs

import vnc_model
from wing_circuit import wing_tables, STEER, WBF

import sys
sys.path.insert(0, "../fly-flight-sim")
import fly_flight2 as FF

prefs.codegen.target = "numpy"

DT_WIN = 0.002
T_TOTAL = 1.5
R_HAL, GYRO = 250.0, 3.0
TAU_STEER = 0.012          # 操舵筋の単収縮時定数 (fast)
G_AMP, G_BIAS, G_PITCH = 0.5, 0.4, 0.3   # 較正定数(3個)

P = {k: float(v) for k, v in dict(np.load(
    "../fly-flight-sim/outputs/hover_params.npz")).items()}
TH = np.deg2rad(47.5)
Q0 = [np.cos(-TH / 2), 0, np.sin(-TH / 2), 0]


def run(reflex_on, T=T_TOTAL, video=None):
    power_ids, steer, hal_L, hal_R = wing_tables()
    steer_all = steer
    tonic = {int(b): 8.5 for b in power_ids}
    excit = {int(b): 4.0 for b in steer_all.bodyid}   # 操舵MNの高入力抵抗 (b1の高感受性)
    ctrl = hal_L + hal_R
    net, mon, ids, idx, pg = vnc_model.make_network(
        [], r_stim_hz=0, sensory_bodyids=ctrl, tonic_mv=tonic,
        excitability=excit)
    kept = [b for b in ctrl if b in idx]
    pos_of = {b: i for i, b in enumerate(kept)}
    sl_L = np.array([pos_of[b] for b in hal_L if b in pos_of])
    sl_R = np.array([pos_of[b] for b in hal_R if b in pos_of])
    steer = steer[steer.bodyid.isin(idx)].reset_index(drop=True)
    sni = steer.bodyid.map(idx).values
    s_tgt = steer.target.values
    s_side = steer.side.astype(str).str[:1].values

    import os
    cwd = os.getcwd()
    os.chdir("../fly-flight-sim")
    try:
        m = FF.build_model()
    finally:
        os.chdir(cwd)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                     "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]}
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)

    renderer, frames, next_f = None, [], 0.0
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = 5.0, -10, 100

    rates = np.zeros(len(kept))
    prev_count = np.zeros(len(ids))
    act = np.zeros(len(steer))
    base_act = None
    warm = []
    R = np.zeros(9)
    zs, ups = [], []
    n_win = int(T / DT_WIN)
    n_phys = int(DT_WIN / dt)
    for wi in range(n_win):
        t = wi * DT_WIN
        om = d.qvel[3:6]
        ph = 2 * np.pi * WBF * t
        base = R_HAL * 0.5 * (1 + np.cos(ph))
        if reflex_on:
            mod_L = 1 + GYRO * (+0.05 * om[0] + 0.05 * om[1] * np.cos(ph)
                                + 0.05 * om[2] * np.sin(ph))
            mod_R = 1 + GYRO * (-0.05 * om[0] + 0.05 * om[1] * np.cos(ph)
                                - 0.05 * om[2] * np.sin(ph))
        else:
            mod_L = mod_R = 1.0
        rates[sl_L] = np.clip(base * mod_L, 0, 400)
        rates[sl_R] = np.clip(base * mod_R, 0, 400)
        pg.rates = rates * Hz
        net.run(2 * ms)
        count = mon.count[:]
        d_spk = (count - prev_count)[sni]
        prev_count = count.copy()
        act += DT_WIN * (d_spk / DT_WIN - act) / TAU_STEER
        # 基準活動 (最初の0.2sの平均) からの相対変調
        if t < 0.1:
            warm.append(act.copy())
            rel = np.zeros(len(steer))
        else:
            if base_act is None:
                base_act = np.mean(warm, axis=0) + 1e-3
            rel = (act - base_act) / (base_act + 5.0)
        # Melis準拠の写像 (筋群→翅運動学)
        def group(side, targets, sign=+1):
            mask = (np.isin(s_tgt, targets)) & (s_side == side)
            return sign * rel[mask].sum()
        damp_L = group("L", ["b1", "b2"]) - group("L", ["b3", "i1"])
        damp_R = group("R", ["b1", "b2"]) - group("R", ["b3", "i1"])
        bias_c = 0.5 * (group("L", ["hg1", "hg2", "hg3", "hg4"])
                        + group("R", ["hg1", "hg2", "hg3", "hg4"]))
        pitch_c = 0.5 * (group("L", ["iii1", "iii3"])
                         + group("R", ["iii1", "iii3"]))
        ampL = np.clip(1.0 + G_AMP * damp_L, 0.6, 1.5)
        ampR = np.clip(1.0 + G_AMP * damp_R, 0.6, 1.5)
        u1 = np.clip(G_BIAS * bias_c, -0.4, 0.4)
        u3 = np.clip(G_PITCH * pitch_c, -0.3, 0.3)
        # 翅駆動 (ホバリング解の運動学に変調を乗せる)
        env0 = min(t / 0.04, 1.0)
        for k in range(n_phys):
            tt = t + k * dt
            ph2 = 2 * np.pi * P["freq"] * tt
            s = np.sin(ph2)
            kk = max(P["sharp"], 1e-3)
            rot = np.tanh(kk * np.cos(ph2 + P["phase"])) / np.tanh(kk)
            d.ctrl[:] = 0
            d.ctrl[aid["wing_yaw_left"]] = env0 * ampL * (P["yaw_amp"] * s + u1)
            d.ctrl[aid["wing_yaw_right"]] = env0 * ampR * (P["yaw_amp"] * s + u1)
            d.ctrl[aid["wing_pitch_left"]] = env0 * ampL * (-P["pitch_amp"] * rot + P["pitch_bias"] + u3)
            d.ctrl[aid["wing_pitch_right"]] = env0 * ampR * (-P["pitch_amp"] * rot + P["pitch_bias"] + u3)
            d.ctrl[aid["wing_roll_left"]] = env0 * P["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = env0 * P["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zs.append(d.qpos[2])
        ups.append(R[8])
        if renderer and t * 10 >= next_f:
            cam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
            next_f += 1.0 / 30
    if renderer and frames:
        import imageio
        imageio.mimsave(video, frames, fps=30)
    zs, ups = np.array(zs), np.array(ups)
    return dict(surv=len(zs) * DT_WIN, z_end=float(zs[-1]) if len(zs) else 0,
                up_mean=float(np.mean(ups)) if len(ups) else 0,
                up_min=float(np.min(ups)) if len(ups) else 0)


if __name__ == "__main__":
    print("A) 反射OFF:", run(False))
    print("B) コネクトーム反射ON:", run(True, video="outputs/connectome_flight.mp4"))
