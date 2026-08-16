#!/usr/bin/env python
"""統合26 LoopA: 前進飛行 — 推力ベクトル制御による水平移動の獲得。

原理 (実物のハエ・マルチコプタと同一): 姿勢目標を進行方向に傾ける
→ 姿勢制御が体を傾ける → 推力の水平成分が生じ前進する。
姿勢の「目標」ZT_W を水平指令 (ax, ay) だけ傾けた単位ベクトルに置換:
  zt_cmd = normalize(ZT_W + [ax, ay, 0])
外側ループ: 位置P-D制御 a = clip(KP·(p_tgt − p) − KD·v, ±A_MAX)。

飛行基盤は ES-FULL (真値状態) — 統合25で特性評価済みの工学層。
本ループの主眼は「行動の土台 (移動能力)」の獲得。

mode=test : 定傾斜→速度計測、ウェイポイント到達試験
"""
import sys
import numpy as np
import mujoco

import phase_reflex as PR
import optics as OPT

N_X, N_U = 7, 5
U_SCALE = np.array([0.5, 0.5, 0.4, 0.4, 0.3])
KP_POS = 0.015     # 位置→傾き [rad/単位]
KD_POS = 0.010     # 速度制動
A_MAX = 0.10       # 最大傾き指令 [rad]
Z_HOVER = 12.0


def fly(T=4.0, waypoint=(8.0, 0.0), fixed_tilt=None, video=None,
        z_tgt=Z_HOVER, ctrl_hook=None, m=None, d=None, log=None,
        du1=0.0, du2=0.0):
    """ウェイポイントへ飛ぶ。(到達距離, 生存s, 直立度) を返す。
    ctrl_hook(t, d) -> (ax, ay, z_tgt, wings_on) を返すと外部から誘導できる"""
    env = PR._fly_env()
    b_pol = env["b_pol"]
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    if m is None:
        m = OPT.get_model()
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[2] = Z_HOVER
        d.qpos[3:7] = Q0
        mujoco.mj_forward(m, d)
    ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    K = ckp["theta"][:35].reshape(5, 7).copy()
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
    dtp = m.opt.timestep
    renderer, frames, next_f = None, [], 0.0
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        vcam = mujoco.MjvCamera()
        vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        vcam.distance, vcam.elevation, vcam.azimuth = 8.0, -10, 120
    R = np.zeros(9)
    kk = max(Pw["sharp"], 1e-3)
    ups, n_alive = [], 0
    wings_on = True
    t_off = None
    for k in range(int(T / dtp)):
        t = k * dtp
        p = d.qpos[:3]
        v = d.qvel[:3]
        if ctrl_hook is not None:
            ax, ay, z_tgt_now, wings_now = ctrl_hook(t, d)
            if wings_on and not wings_now:
                t_off = t
            wings_on = wings_now
        elif fixed_tilt is not None:
            ax, ay = fixed_tilt
            z_tgt_now = z_tgt
        else:
            ax = np.clip(KP_POS * (waypoint[0] - p[0]) - KD_POS * v[0],
                         -A_MAX, A_MAX)
            ay = np.clip(KP_POS * (waypoint[1] - p[1]) - KD_POS * v[1],
                         -A_MAX, A_MAX)
            z_tgt_now = z_tgt
        zt = ZT_W + np.array([-ax, -ay, 0.0])  # 実測: 推力水平成分は逆符号
        zt /= np.linalg.norm(zt)
        # 目標z軸を傾けた姿勢誤差
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, zt)
        om = d.qvel[3:6]
        x = np.array([(z_tgt_now - p[2]) / 5.0, -v[2] / 30.0,
                      e_b[0], e_b[1],
                      om[0] / 20.0, om[1] / 20.0, om[2] / 20.0])
        u = np.tanh(K @ x + b_pol) * U_SCALE
        u[1] += du1
        u[2] += du2
        amp = np.clip(1.0 + u[0], 0.5, 1.6)
        env0 = min(t / 0.03, 1.0) if wings_on else 0.0
        ph2 = 2 * np.pi * Pw["freq"] * t
        s = np.sin(ph2)
        rot = np.tanh(kk * np.cos(ph2 + Pw["phase"])) / np.tanh(kk)
        eL = env0 * amp
        eR = env0 * amp
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
        if log is not None and k % 200 == 0:
            log.append((t, p[0], p[1], p[2], v[0]))
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.3:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        n_alive = k + 1
        if renderer and t * 10 >= next_f:
            vcam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, camera=vcam)
            frames.append(renderer.render())
            next_f += 1.0 / 30
        if t_off is not None and t - t_off > 0.5:
            break       # 翅停止後は少し見て終了
    if renderer and frames:
        import imageio
        imageio.mimsave(video, frames, fps=30)
    return d, n_alive * dtp, float(np.mean(ups)) if ups else 0.0


def mode_test():
    # 1) 定傾斜→前進速度
    for tilt in [0.1, 0.2]:
        log = []
        d, s, up = fly(T=2.0, fixed_tilt=(tilt, 0.0), log=log)
        vx = np.mean([r[4] for r in log[len(log)//2:]])
        print(f"定傾斜{tilt:.1f}rad: 生存{s:.2f}s 前進速度vx≈{vx:.2f} "
              f"到達x={log[-1][1]:.1f}", flush=True)
    # 2) ウェイポイント (x=+8)
    log = []
    d, s, up = fly(T=4.0, waypoint=(8.0, 0.0), log=log)
    xe = log[-1][1]
    print(f"ウェイポイント(8,0): 生存{s:.2f}s 最終x={xe:.2f} z={log[-1][3]:.1f} "
          f"直立度{up:+.2f}", flush=True)


if __name__ == "__main__":
    mode_test()
