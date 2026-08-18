#!/usr/bin/env python
"""統合32c: 飛行制御はω復号誤差をどこまで許容するか (閾値の定量)。

統合32で実配線の復号誤差0.113 vs シャッフル0.191という差を検出したが、
行動では差が出なかった (筋数削減・大外乱の2通りで確認)。
ならば「制御がどれだけの復号誤差まで耐えるか」を測れば、
差が行動に出ない理由を定量的に説明できる。

真のω に相対誤差 e の乱数を加えて飛行させ、生存が崩れる閾値を求める。
"""
import sys
import numpy as np
import mujoco
from multiprocessing import Pool
import openloop_hover as OH
import bioflight as BF

TH = np.load("outputs/bioflight_best.npy")
N_X, N_U = 7, 5


def rollout(err, pert_seed=0, T=3.0):
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P, u_trim = OH.unpack(TH[:12])
    K = TH[12:12 + N_U * N_X].reshape(N_U, N_X)
    bb = TH[12 + N_U * N_X:]
    rg = np.random.default_rng(1234 + pert_seed)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(P["sharp"], 1e-3)
    R = np.zeros(9)
    NB = max(int((1.0 / P["freq"]) / dtp), 1)
    buf = np.zeros((NB, 3))
    bi, ssum = 0, np.zeros(3)
    hold = 1.0 / P["freq"]
    t_next = 0.0
    u_cmd = np.array(u_trim, float)
    u = np.array(u_trim, float)
    TAU_TW = 0.00425
    ups, zs, alive = [], [], 0
    for k in range(int(T / dtp)):
        t = k * dtp
        ssum += d.qvel[3:6] - buf[bi]
        buf[bi] = d.qvel[3:6].copy()
        bi = (bi + 1) % NB
        if t >= t_next:
            t_next += hold
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            v = d.qvel[:3]
            ow = ssum / NB
            if err > 0:            # 相対誤差を注入 (復号の不正確さの模擬)
                ow = ow + err * np.linalg.norm(ow) * rg.standard_normal(3)
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(u_trim + np.tanh(K @ x + bb) * 0.35, -0.55, 0.55)
        u += dtp * (u_cmd - u) / TAU_TW
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0)
        ph2 = 2 * np.pi * P["freq"] * t
        s = np.sin(ph2)
        rot = np.tanh(kk * np.cos(ph2 + P["phase"])) / np.tanh(kk)
        e = env0 * amp
        d.ctrl[:] = 0
        d.ctrl[aid["wing_yaw_left"]] = e * (P["yaw_amp"] * s + u[1] + u[2])
        d.ctrl[aid["wing_yaw_right"]] = e * (P["yaw_amp"] * s + u[1] - u[2])
        d.ctrl[aid["wing_pitch_left"]] = e * (-P["pitch_amp"] * rot
                                              + P["pitch_bias"] + u[3] + u[4])
        d.ctrl[aid["wing_pitch_right"]] = e * (-P["pitch_amp"] * rot
                                               + P["pitch_bias"] + u[3] - u[4])
        d.ctrl[aid["wing_roll_left"]] = e * P["roll_amp"] * np.sin(2 * ph2)
        d.ctrl[aid["wing_roll_right"]] = e * P["roll_amp"] * np.sin(2 * ph2)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5 or d.qpos[2] > 40:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        zs.append(d.qpos[2])
        alive = k + 1
    return (alive * dtp, float(np.mean(ups)) if ups else 0.0,
            float(np.mean(np.abs(np.array(zs) - 12.0))) if zs else 20.0)


def job(a):
    return rollout(a[0], pert_seed=a[1])


if __name__ == "__main__":
    ERRS = [0.0, 0.11, 0.19, 0.4, 0.8, 1.5, 3.0, 6.0]
    jobs = [(e, s) for e in ERRS for s in range(4)]
    with Pool(min(len(jobs), 16)) as p:
        res = p.map(job, jobs)
    k = 0
    print("復号誤差(相対) → 飛行 (4外乱平均)", flush=True)
    for e in ERRS:
        v = res[k:k + 4]
        k += 4
        sv = np.array([x[0] for x in v])
        up = np.array([x[1] for x in v])
        ze = np.array([x[2] for x in v])
        note = ""
        if abs(e - 0.11) < 1e-6:
            note = "  ← 実配線の実測値"
        if abs(e - 0.19) < 1e-6:
            note = "  ← シャッフル平均"
        print(f"  誤差{e:4.2f}: 生存{sv.mean():.2f}s 直立{up.mean():+.2f} "
              f"高度誤差{ze.mean():.1f}{note}", flush=True)
    print("DONE", flush=True)
