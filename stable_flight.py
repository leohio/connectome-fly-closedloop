#!/usr/bin/env python
"""統合24: 作動経路の再設計 → コネクトーム閉ループの安定飛行。

統合23の最終診断: 神経推定は完璧、失敗の残因は作動経路
(2ms窓+τ100ms低域+ヒンジ変調 vs ES-FULLの50μs即時線形FB)。
本統合は姿勢推定の「使い先」を変更する:
  推定 êb (2ms更新+低域τ) → ES学習済み u1-u4 線形チャネル
  (物理ステップ毎に u=tanh(Kx+b) を評価。x の姿勢2成分だけ êb、
   z/vz/ω は基盤側の真値 = ωダンピング基盤は統合22の帰結)
これは「ESポリシーの姿勢入力を、真の値からコネクトーム視覚の推定値に
差し替える」ことに等しい — 視覚の代理(ES)を本物の視覚で置換する最終形。

mode=oracle : 真値姿勢でτ耐性を測定 (神経への帯域要求仕様)
mode=neuro  : 集団デコード接続の単発試験
mode=main   : 最終検定 (REAL / SHUF-adapt / SHUF-fix, n=6)
mode=long   : T=10s 長時間安定の実証
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, prefs

import phase_reflex as PR
import brain_visual as BV
import vision_pop as VP

prefs.codegen.target = "numpy"

DT_W = 0.002
TAU_F = 0.15    # 相補フィルタ: 視覚によるドリフト補正の時定数


def trial(est="oracle", tau=0.02, shuffle=None, seed=0, transfer_w=None,
          att_scale=1.0, T=3.0, video=None):
    """(生存s, 直立度, z平均) を返す。est: oracle / neuro / none"""
    from brian2 import ms as _ms
    env = PR._fly_env()
    b_pol, U_SCALE = env["b_pol"], env["U_SCALE"]
    Pw, Q0, ZT_W, m = env["Pw"], env["Q0"], env["ZT_W"], env["m"]
    ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    K = ckp["theta"][:5 * 7].reshape(5, 7).copy()
    K[:, 2:4] *= att_scale
    if est == "none":
        K[:, 2:4] = 0.0
    if est in ("neuro", "fusion"):
        net, mon, pg, ssign, is_oc, readout = VP.build(shuffle, seed)
        r0, t_r, t_p, s2r, s2p = VP.calibrate(net, mon, pg, ssign, is_oc,
                                              readout)
        if transfer_w is not None:
            t_r, t_p, s2r, s2p = transfer_w
        prev_c = mon.count[:].copy()
        f_rate = r0.copy()
    eb_est = np.zeros(2)
    d = mujoco.MjData(m)
    dtp = m.opt.timestep
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
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
    R = np.zeros(9)
    ups, zs, n_alive = [], [], 0
    kk = max(Pw["sharp"], 1e-3)
    for wi in range(int(T / DT_W)):
        t = wi * DT_W
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        zc = np.array([R[2], R[5], R[8]])
        e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
        if est == "oracle":
            target = np.array([e_b[0], e_b[1]])
        elif est in ("neuro", "fusion"):
            om = d.qvel[3:6]
            pg.rates = BV.vis_rates(e_b[0], e_b[1], om[0], om[1],
                                    ssign, is_oc) * Hz
            net.run(DT_W * 1000 * _ms)
            c = mon.count[:].copy()
            r_now = (c - prev_c)[readout] / DT_W
            prev_c = c
            f_rate += DT_W * (r_now - f_rate) / max(tau, DT_W)
            target = np.array([VP.decode(f_rate, r0, t_r, s2r),
                               VP.decode(f_rate, r0, t_p, s2p)])
        else:
            target = np.zeros(2)
        if est == "fusion":
            om = d.qvel[3:6]
            # 高速: ω積分で姿勢伝搬 (d eb/dt = [-ωx, -ωy], 実測R²=0.95)
            eb_est = eb_est + DT_W * np.array([-om[0], -om[1]])
            # 低速: 視覚推定がドリフトを補正
            eb_est += (DT_W / TAU_F) * (target - eb_est)
        elif est == "oracle" and tau > 0:
            eb_est += DT_W * (target - eb_est) / tau
        else:
            eb_est = target
        for kp in range(int(DT_W / dtp)):
            tt = t + kp * dtp
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            omp = d.qvel[3:6]
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -d.qvel[2] / 30.0,
                          eb_est[0], eb_est[1],
                          omp[0] / 20.0, omp[1] / 20.0, omp[2] / 20.0])
            u = np.tanh(K @ x + b_pol) * U_SCALE
            amp = np.clip(1.0 + u[0], 0.5, 1.6)
            env0 = min(tt / 0.03, 1.0)
            ph2 = 2 * np.pi * Pw["freq"] * tt
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
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        zs.append(d.qpos[2])
        n_alive = wi + 1
        if renderer and t * 10 >= next_f:
            cam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
            next_f += 1.0 / 30
    if renderer and frames:
        import imageio
        imageio.mimsave(video, frames, fps=30)
    return (n_alive * DT_W, float(np.mean(ups)) if ups else 0.0,
            float(np.mean(zs)) if zs else 0.0)


def mode_oracle():
    s, up, z = trial(est="none")
    print(f"基盤のみ(姿勢なし): 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    for tau in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        s, up, z = trial(est="oracle", tau=tau)
        print(f"ORACLE τ={tau*1000:.0f}ms: 生存{s:.2f}s 直立度{up:+.2f} "
              f"z={z:.1f}", flush=True)


def mode_neuro():
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    for rep in range(3):
        s, up, z = trial(est="neuro", tau=tau)
        print(f"NEURO τ={tau*1000:.0f}ms rep{rep}: 生存{s:.2f}s "
              f"直立度{up:+.2f} z={z:.1f}", flush=True)


def mode_main():
    NREP = 6
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    print(f"=== 統合24最終検定(fusion) (n={NREP}, τ={tau*1000:.0f}ms, T=3s) ===",
          flush=True)
    s, up, z = trial(est="none")
    print(f"{'基盤のみ':12s} 生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    net, mon, pg, ssign, is_oc, readout = VP.build(None, 0)
    _, t_r, t_p, s2r, s2p = VP.calibrate(net, mon, pg, ssign, is_oc, readout)
    real_w = (t_r, t_p, s2r, s2p)
    rows = [("BASE", [s], [up])]
    for name, sh, seed, tw in [("VIS-REAL", None, 0, None),
                               ("SHUF-adapt s0", "all", 0, None),
                               ("SHUF-fix s0", "all", 0, real_w),
                               ("SHUF-fix s1", "all", 1, real_w)]:
        ss, uu = [], []
        for rep in range(NREP):
            s, up, z = trial(est="fusion", tau=tau, shuffle=sh, seed=seed,
                             transfer_w=tw)
            ss.append(s); uu.append(up)
            print(f"  {name} rep{rep}: 生存{s:.2f}s 直立度{up:+.2f} z={z:.1f}",
                  flush=True)
        print(f"{name:14s} 生存{np.mean(ss):.2f}±{np.std(ss):.2f}s "
              f"直立度{np.mean(uu):+.3f}±SEM{np.std(uu)/np.sqrt(NREP):.3f}",
              flush=True)
        rows.append((name, ss, uu))
    np.savez("outputs/stable_main.npz", rows=np.array(rows, dtype=object))


def mode_long():
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    s, up, z = trial(est="fusion", tau=tau, T=10.0,
                     video="outputs/stable_vision_flight.mp4")
    print(f"LONG T=10s: 生存{s:.2f}s 直立度{up:+.2f} z={z:.1f}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "oracle"
    {"oracle": mode_oracle, "neuro": mode_neuro, "main": mode_main,
     "long": mode_long}[mode]()
