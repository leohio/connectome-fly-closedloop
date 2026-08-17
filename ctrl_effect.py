#!/usr/bin/env python
"""統合29 LoopF: 操舵チャネル u1-u4 の制御有効性行列 B を測る。

u_k に一定オフセットを与えたときの機体角加速度 (roll,pitch,yaw) を測定し、
B[k] = d(角加速度)/du_k を得る。これが分かれば、望むトルクを出すための
チャネル配分を解析的に決められる (盲目的ESの代わり)。
"""
import numpy as np
import mujoco
import phase_reflex as PR
import connectome_fastloop as CF

T = 0.25


def run(u_off):
    env = PR._fly_env()
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    m = CF.OPT.get_model()
    aid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                     "wing_yaw_right", "wing_roll_right", "wing_pitch_right"]}
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    NB = len(CF.PHIS)
    w_hist = []
    n = int(T / CF.DT_N)
    for kn in range(n):
        tn = kn * CF.DT_N
        phc = (tn * Pw["freq"]) % 1.0
        i = min(int(phc * NB), NB - 1)
        u14 = CF.Z["recon"][i] + u_off       # 蒸留FF + 一定オフセット
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
                                                  + Pw["pitch_bias"] + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = e * (-Pw["pitch_amp"] * rot
                                                   + Pw["pitch_bias"] + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        if tn > 0.10:
            w_hist.append(d.qvel[3:6].copy())
    if len(w_hist) < 20:
        return None
    W = np.array(w_hist)
    t = np.arange(len(W)) * CF.DT_N
    # 羽ばたき反動の振動を含むので線形回帰の傾き=平均角加速度
    A = np.vstack([t, np.ones_like(t)]).T
    sl, _ = np.linalg.lstsq(A, W, rcond=None)[0]
    return sl


if __name__ == "__main__":
    base = run(np.zeros(4))
    print(f"基準 角加速度 (roll,pitch,yaw) = {np.round(base, 1)}", flush=True)
    D = 0.25
    B = []
    for k in range(4):
        off = np.zeros(4)
        off[k] = D
        p = run(off)
        off[k] = -D
        mn = run(off)
        if p is None or mn is None:
            print(f"u{k+1}: 墜落で測定不能", flush=True)
            B.append(np.zeros(3))
            continue
        b = (p - mn) / (2 * D)
        B.append(b)
        print(f"u{k+1}: dω̇/du = {np.round(b, 1)}  "
              f"(+{D}: {np.round(p,1)} / -{D}: {np.round(mn,1)})", flush=True)
    B = np.array(B)
    np.save("outputs/ctrl_B.npy", B)
    print("\n制御有効性行列 B (行=u1..u4, 列=roll,pitch,yaw):", flush=True)
    print(np.round(B, 1), flush=True)
    print("DONE", flush=True)
