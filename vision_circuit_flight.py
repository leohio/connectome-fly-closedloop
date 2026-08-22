#!/usr/bin/env python
"""統合37: 脳視覚経路を飛行ループに実接続する。

これまで姿勢入力 e_b は物理状態から直接計算していた (視覚のスタブ)。
ここでは FlyWire実測の視覚サブ回路 (オセリ273→OCG20→DN23, VS/HS38→中継→DN)
をbrian2で回し、その整合フィルタ復号を制御則の姿勢入力にする。

構成: 腹髄net (ハルテア→実配線→MN→位相復号ω) + 脳視覚net (姿勢) + flybody
- 視覚は2ms窓で駆動、レート平滑 τ_DN=30ms → 遅延24ms・ノイズ1.1° (実測特性、
  実バエのオセリ遅延20-30msと同域)
- ωチャネルは従来どおり腹髄回路の復号 (速い)、姿勢チャネルは脳視覚 (遅い)
  — ハルテアと視覚の実バエの分担そのもの
"""
import sys
import json
import numpy as np
import mujoco
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC
import openloop_hover as OH
import connectome_bioflight as CB
import vision_pop as VP
import brain_visual as BV

DT_W = 0.002
TAU_DN = 0.03


def fly(mode="vision", K_src="learned", W_src="calibrated", T=10.0,
        pert_seed=0):
    """mode: vision=脳視覚でe_b / stub=従来(直接計算)"""
    if K_src == "learned":
        res = json.load(open("outputs/reward_K.json"))
        sel = json.load(open("outputs/reward_K_selected.json"))["selected_seed"]
        th = np.array([r for r in res if r["seed"] == sel][0]["theta"])
        CB.K_POL = th[:CB.N_U * CB.N_X].reshape(CB.N_U, CB.N_X)
        CB.B_POL = th[CB.N_U * CB.N_X:]
    if W_src == "learned":
        import plastic_readout as P
        d0 = P.episodes(rng_seed=0, n_train=120)
        Wd, _ = P.learn(d0, rng_seed=0, checkpoints=[120])
        dec = (Wd, d0["names"], d0["phi0"])
    else:
        dec = CB.calibrate()
    Sg, names, phi0 = dec
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    P0, U_TRIM = CB.P0, CB.U_TRIM
    net, mon, pg, pref, side, st_idx, n = PR.setup(**LC.SETUP_KW)
    idx_of = {j: st_idx[j] for j in names if j in st_idx}
    zc = {j: 0j for j in names}
    prev = 0
    shift_prev = np.zeros(len(pref))
    if mode == "vision":
        VP.TAU_DN = TAU_DN
        vnet, vmon, vpg, ssign, is_oc, readout = VP.build()
        r0, t_r, t_p, s2r, s2p = VP.calibrate(vnet, vmon, vpg, ssign, is_oc,
                                              readout)
        vprev = vmon.count[:].copy()
        f_rate = r0.copy()
        print(f"視覚較正完了 (読出し{len(readout)}ニューロン)", flush=True)
    eb_est = np.zeros(2)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
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
    t_next, t_vis = 0.0, 0.0
    ups, zs, alive = [], [], 0
    n_sub = max(int(round(CF.DT_N / dtp)), 1)
    NV = max(int(DT_W / CF.DT_N), 1)
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        omsum += d.qvel[3:6] - ombuf[ombi]
        ombuf[ombi] = d.qvel[3:6].copy()
        ombi = (ombi + 1) % NBOX
        o_slow = omsum / NBOX
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
        if mode == "vision" and kn % NV == 0:
            # 眼は実際の地平線 (=真の姿勢) を見る。復号が推定を返す
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            omv = d.qvel[3:6]
            vpg.rates = BV.vis_rates(e_b[0], e_b[1], omv[0], omv[1],
                                     ssign, is_oc) * Hz
            vnet.run(DT_W * 1000 * _ms)
            c = vmon.count[:].copy()
            f_rate += DT_W * ((c - vprev)[readout] / DT_W - f_rate) / TAU_DN
            vprev = c
            eb_est = np.array([VP.decode(f_rate, r0, t_r, s2r),
                               VP.decode(f_rate, r0, t_p, s2p)])
        if tn >= t_next:
            t_next += per
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            v = d.qvel[:3]
            eb_use = eb_est if mode == "vision" else e_b[:2]
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          eb_use[0], eb_use[1],
                          om_est[0] / 20.0, om_est[1] / 20.0,
                          om_est[2] / 20.0])
            u_cmd = np.clip(U_TRIM + np.tanh(CB.K_POL @ x + CB.B_POL) * 0.35,
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
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5 or d.qpos[2] > 40:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        zs.append(d.qpos[2])
        alive = kn + 1
    return (alive * CF.DT_N, float(np.mean(ups)) if ups else 0.0,
            float(np.mean(np.abs(np.array(zs) - 12.0))) if zs else 20.0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "vision"
    pert = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    W_src = sys.argv[3] if len(sys.argv) > 3 else "calibrated"
    r = fly(mode=mode, W_src=W_src, pert_seed=pert)
    print(f"{mode}/W={W_src}(seed{pert}): 生存{r[0]:.2f}s 直立{r[1]:+.2f} "
          f"高度誤差{r[2]:.1f}", flush=True)
