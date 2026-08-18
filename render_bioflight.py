#!/usr/bin/env python
"""統合31: コネクトーム駆動の安定ホバリングを動画にする。

実MANCハルテア求心性 → 実配線 → 操舵MN (1周期1発) → 筋活動位相 → 較正復号ω
→ 1羽ばたき1回の帰還 → 筋単収縮 → 翅運動学、を回しながらレンダリングする。
"""
import sys
import numpy as np
import mujoco
import imageio
from brian2 import ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC
import openloop_hover as OH
import connectome_bioflight as CB

FPS = 60


def render(src="circuit", T=6.0, out="outputs/bioflight.mp4", pert_seed=0):
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P0, U_TRIM = CB.P0, CB.U_TRIM
    K_POL, B_POL = CB.K_POL, CB.B_POL
    use_net = src == "circuit"
    if use_net:
        Sg, names, phi0 = CB.calibrate()
        print(f"較正完了: {len(names)}筋", flush=True)
        net, mon, pg, pref, side, st_idx, n = PR.setup(**LC.SETUP_KW)
        idx_of = {j: st_idx[j] for j in names if j in st_idx}
        zc = {j: 0j for j in names}
        prev = 0
        shift_prev = np.zeros(len(pref))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, 480, 640)
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
    t_next, next_f = 0.0, 0.0
    frames = []
    n_sub = max(int(round(CF.DT_N / dtp)), 1)
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        omsum += d.qvel[3:6] - ombuf[ombi]
        ombuf[ombi] = d.qvel[3:6].copy()
        ombi = (ombi + 1) % NBOX
        o_slow = omsum / NBOX
        if use_net:
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
        if tn >= t_next:
            t_next += per
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            v = d.qvel[:3]
            ow = om_est if use_net else o_slow
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(U_TRIM + np.tanh(K_POL @ x + B_POL) * 0.35,
                            -0.55, 0.55)
        for _ in range(n_sub):
            t = d.time
            u += dtp * (u_cmd - u) / CB.TAU_TW
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
    imageio.mimsave(out, frames, fps=FPS, quality=8)
    print(f"保存 {out} ({len(frames)}フレーム, {d.time:.2f}s)", flush=True)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "circuit"
    render(src=src, out=f"outputs/bioflight_{src}.mp4")
