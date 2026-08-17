#!/usr/bin/env python
"""統合28 LoopA: 高速ωループの要求仕様の精密測定。

統合25の帰属: ωを3ms平滑→即墜落 / 周期平均→墜落。だが生物の実装可能領域は
「周期1回サンプル・約1周期遅延・筋収縮フィルタ」。平滑・平均と
サンプル&ホールド(S&H)は周波数特性が違う (S&Hは中帯域を位相遅れ小で保持)。
本実験: ωの供給方式を離散化の梯子で変え、ES-FULLホバーの生存を測る。
  A) 連続 (基準)
  B) S&H 1回/周期 (瞬時値、位相φs固定)
  C) S&H 2回/周期
  D) S&H 1回/周期 + 1周期遅延 (実回路の遅延相当)
  E) S&H 1回/周期 + 筋収縮フィルタ (τ=8.5ms fast筋, Azevedo実測)
生物実装可能性 = D/E が飛ぶかどうかで決まる。
"""
import numpy as np
import mujoco
import phase_reflex as PR
import optics as OPT

WBF_P = None


def trial(mode="cont", phi_s=0.25, T=3.0, tau_tw=0.0085):
    env = PR._fly_env()
    b_pol = env["b_pol"]
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    m = OPT.get_model()
    ck = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    K = ck["theta"][:35].reshape(5, 7)
    US = np.array([0.5, 0.5, 0.4, 0.4, 0.3])
    aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                     "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]}
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    om_held = np.zeros(3)
    om_q = []          # 遅延バッファ (1周期 = 1/freq)
    om_f = np.zeros(3)  # 筋収縮フィルタ状態
    last_cyc = -1
    last_half = -1
    alive = 0
    ups = []
    delay_steps = int(round(1.0 / Pw["freq"] / dtp))
    for k in range(int(T / dtp)):
        t = k * dtp
        om_now = d.qvel[3:6].copy()
        phc = (t * Pw["freq"]) % 1.0
        cyc = int(t * Pw["freq"])
        if mode == "cont":
            om_used = om_now
        elif mode == "sh1":
            if cyc != last_cyc and phc >= phi_s:
                om_held = om_now.copy()
                last_cyc = cyc
            om_used = om_held
        elif mode == "sh2":
            half = int(t * Pw["freq"] * 2)
            if half != last_half:
                om_held = om_now.copy()
                last_half = half
            om_used = om_held
        elif mode == "sh1_delay":
            om_q.append(om_now.copy())
            om_del = om_q[0] if len(om_q) <= delay_steps else om_q[-delay_steps]
            if len(om_q) > delay_steps + 2:
                om_q.pop(0)
            if cyc != last_cyc and phc >= phi_s:
                om_held = om_del.copy()
                last_cyc = cyc
            om_used = om_held
        elif mode == "sh1_twitch":
            if cyc != last_cyc and phc >= phi_s:
                om_held = om_now.copy()
                last_cyc = cyc
            om_f += dtp * (om_held - om_f) / tau_tw
            om_used = om_f
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
        v = d.qvel[:3]
        x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                      e_b[0], e_b[1], om_used[0] / 20.0, om_used[1] / 20.0,
                      om_used[2] / 20.0])
        u = np.tanh(K @ x + b_pol) * US
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0)
        ph2 = 2 * np.pi * Pw["freq"] * t
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
        alive = k + 1
    return alive * dtp, float(np.mean(ups)) if ups else 0


if __name__ == "__main__":
    for mode, label in [("cont", "A) 連続(基準)"),
                        ("sh2", "C) S&H 2回/周期"),
                        ("sh1", "B) S&H 1回/周期"),
                        ("sh1_twitch", "E) S&H1+筋収縮τ8.5ms"),
                        ("sh1_delay", "D) S&H1+1周期遅延")]:
        s, up = trial(mode=mode)
        print(f"{label:24s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
