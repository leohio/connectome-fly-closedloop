#!/usr/bin/env python
"""統合39b: 生得高度反射の設計 — 地上0.2から上昇→巡航→降下→接地直前まで。

学習Kの高度チャネルは弱い (誤差~7)。実バエは腹側オプティックフローによる
高度調節反射を持つため、これを生得反射としてu[0](振幅)に加算する:
  u_alt = KZ·tanh((z_t - z)/ZSCALE) - KV·vz/VSCALE
安価ループ (回路模型感覚) で上昇・保持・降下が成立するゲインを探す。
"""
import sys
import numpy as np
import mujoco
import reward_K as RK
import bioflight as BF
RK.BF = BF
import openloop_hover as OH
import json

KZ, ZS, KVv, VS = 0.22, 3.0, 0.10, 10.0


def seq_rollout(kz=KZ, kv=KVv, T=9.0, show=False):
    import connectome_bioflight as CB
    res = json.load(open("outputs/reward_K.json"))
    sel = json.load(open("outputs/reward_K_selected.json"))["selected_seed"]
    th_l = np.array([r for r in res if r["seed"] == sel][0]["theta"])
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P, u_trim = OH.unpack(RK.INNATE)
    K = th_l[:RK.BF.N_U * RK.BF.N_X].reshape(RK.BF.N_U, RK.BF.N_X)
    bb = th_l[RK.BF.N_U * RK.BF.N_X:]
    rgn = np.random.default_rng(7)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 2.0   # 離陸直後を模擬 (脚支持は実シーケンス側)
    d.qpos[3:7] = Q0
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(P["sharp"], 1e-3)
    NB = max(int((1.0 / P["freq"]) / dtp), 1)
    buf = np.zeros((NB, 3)); bi = 0; ssum = np.zeros(3)
    hold = 1.0 / P["freq"]
    t_next = 0.0
    u_cmd = np.array(u_trim, float); u = np.array(u_trim, float)
    om_sen = np.zeros(3); eb_sen = np.zeros(2)
    R = np.zeros(9)
    zlog = []
    TAU_TW, TAU_DEC, TAU_VIS = 0.00425, 0.012, 0.03
    for k in range(int(T / dtp)):
        t = k * dtp
        ssum += d.qvel[3:6] - buf[bi]; buf[bi] = d.qvel[3:6].copy()
        bi = (bi + 1) % NB
        # フェーズ: 上昇→巡航→降下
        if t < 4.0:
            z_t = min(2.0 + 4.0 * t, 12.0)
        elif t < 6.0:
            z_t = 12.0
        else:
            z_t = max(12.0 - 2.0 * (t - 6.0), 0.1)
        if t >= t_next:
            t_next += hold
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zc = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zc, ZT_W)
            v = d.qvel[:3]
            o_slow = ssum / NB
            tgt = RK.SENSOR_G * np.clip(o_slow, -12, 12) + RK.SENSOR_BIAS
            om_sen += hold * (tgt - om_sen) / TAU_DEC
            ow = om_sen + rgn.normal(0, 1.9, 3)
            tgt_eb = 0.63 * np.tanh(1.2 * e_b[:2])
            eb_sen += hold * (tgt_eb - eb_sen) / TAU_VIS
            ebu = eb_sen + rgn.normal(0, 0.019, 2)
            x = np.array([(z_t - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          ebu[0], ebu[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(u_trim + np.tanh(K @ x + bb) * 0.35, -0.55, 0.55)
            # 生得高度反射 (腹側オプティックフローのモデル)
            u_cmd[0] = np.clip(u_cmd[0] + kz * np.tanh((z_t - d.qpos[2]) / ZS)
                               - kv * v[2] / VS, -0.6, 0.6)
        u += dtp * (u_cmd - u) / TAU_TW
        amp = np.clip(1.0 + u[0], 0.5, 1.7)
        env0 = 1.0
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
        if t < 0.15:      # 脚支持の代理: 離陸スプールアップ中は保持
            d.qpos[0:3] = [0, 0, 2.0]
            d.qvel[0:3] = 0
        if not np.isfinite(d.qpos[2]) or d.qpos[2] > 40:
            return None
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        up = R[8]
        zlog.append((t, d.qpos[2], up))
        if d.qpos[2] < 0.05 and t < 5.5:
            return ("墜落", t, np.array(zlog))
        if t > 6.0 and d.qpos[2] < 0.15:
            return ("接地", t, np.array(zlog))
    return ("完走", T, np.array(zlog))


if __name__ == "__main__":
    for kz, kv in [(0.0, 0.0), (0.15, 0.08), (0.22, 0.10), (0.3, 0.15)]:
        r = seq_rollout(kz, kv)
        if r is None:
            print(f"kz={kz} kv={kv}: 発散")
            continue
        st, t, zl = r
        m34 = zl[(zl[:, 0] > 3.5) & (zl[:, 0] < 6.0)]
        up_end = zl[-1, 2]
        print(f"kz={kz:4.2f} kv={kv:4.2f}: {st} t={t:.2f}s "
              f"巡航z={m34[:,1].mean() if len(m34) else 0:.1f}"
              f"±{m34[:,1].std() if len(m34) else 0:.1f} "
              f"最大z={zl[:,1].max():.1f} 終端直立={up_end:+.2f}", flush=True)
