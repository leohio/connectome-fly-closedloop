#!/usr/bin/env python
"""統合30: 安定飛行に必要な作動帯域を定量する。

完全な感覚 (真のω全帯域) を与えたうえで、翅への指令 u の更新レートだけを
1羽ばたきあたり N 回に制限する。N をいくつまで下げられるかが、
「神経筋系で実装可能か」の判定になる。
操舵MNは1周期1発 (N=1相当)、単収縮は4.25ms (≈1周期) である。
"""
import sys
import numpy as np
import mujoco
import bio_train as BT

CK = np.load("../fly-flight-sim/outputs/hover_policy.npz")["theta"]
K = CK[:35].reshape(5, 7)
B = CK[35:40]


def run(n_per_stroke, T=3.0, pert_seed=None):
    E = BT.env()
    m, Pw, Q0, ZT_W, US, aid = (E["m"], E["Pw"], E["Q0"], E["ZT_W"],
                                E["US"], E["aid"])
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    R = np.zeros(9)
    period = 1.0 / Pw["freq"]
    hold = period / n_per_stroke if n_per_stroke > 0 else dtp
    u = np.zeros(5)
    t_next = 0.0
    ups, alive = [], 0
    for k in range(int(T / dtp)):
        t = k * dtp
        if n_per_stroke <= 0 or t >= t_next:      # 指令の更新タイミング
            t_next += hold
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            v = d.qvel[:3]
            ow = d.qvel[3:6]
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u = np.tanh(K @ x + B) * US
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
    return alive * dtp


def _job(j):
    return run(j[0], pert_seed=j[1])


if __name__ == "__main__":
    from multiprocessing import Pool
    RATES = [0, 64, 32, 16, 8, 4, 2, 1]
    jobs = [(r, s) for r in RATES for s in range(3)]
    with Pool(12) as p:
        res = p.map(_job, jobs)
    k = 0
    for r in RATES:
        v = np.array(res[k:k + 3])
        k += 3
        lab = "連続(制限なし)" if r == 0 else f"{r:2d}回/羽ばたき"
        hz = "" if r == 0 else f" ({r*202.3:.0f}Hz)"
        print(f"{lab:14s}{hz:12s} 生存 平均{v.mean():.2f}s "
              f"({', '.join(f'{x:.2f}' for x in v)})", flush=True)
    print("DONE", flush=True)
