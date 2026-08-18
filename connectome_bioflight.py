#!/usr/bin/env python
"""統合31 最終: 神経筋制約を満たす飛行制御を、実MANCコネクトーム回路で駆動する。

構成 (すべて生物学的制約を満たす):
  実ハルテア求心性 (55本, MANC) → 実配線 (traced-connections) → 操舵MN (1周期1発)
    → 筋活動位相 → 較正復号 ω_est → 1羽ばたき1回の帰還則 → 筋単収縮 τ=4.25ms
    → 翅運動学
禁止事項を守っている: 自分の羽ばたき振動の瞬時値は使わない (ハルテアは1周期
移動平均=周波数選択的復調のみ)。指令更新は1羽ばたき1回。

対照: (a) 真の遅いω (完全な感覚), (b) 回路復号ω, (c) シャッフル配線の回路,
      (d) ω=0 (ハルテア喪失)
"""
import sys
import numpy as np
import mujoco
from brian2 import ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC
import openloop_hover as OH

TH = np.load("outputs/bioflight_best.npy")
P0, U_TRIM = OH.unpack(TH[:12])
N_X, N_U = 7, 5
K_POL = TH[12:12 + N_U * N_X].reshape(N_U, N_X)
B_POL = TH[12 + N_U * N_X:]
PR.WBF = float(P0["freq"])      # ハルテア-翅の機械的周波数結合
PR.C_PHASE = 0.0015             # 生理的な位相シフト感度
TAU_TW = 0.00425                # 筋単収縮 (Azevedo 2020)
PERT_SIG = 1.5                  # 初期外乱の大きさ [rad/s]


def calibrate(shuffle=None, seed=0, n_mus=None):
    """この翅周波数で神経ω復号行列Sと基準位相φ0を較正する。
    n_mus を指定すると位相固定度Rの高い順にその本数だけ使う
    (課題を難しくして配線特異性が行動に現れるかを見るため)"""
    import measure_S as MS
    MS.PR.WBF = PR.WBF
    S, R, base = MS.build(shuffle=shuffle, seed=seed)
    good = [i for i in range(len(MS.MUS)) if R[i] > 0.5]
    if n_mus is not None and len(good) > n_mus:
        good = sorted(good, key=lambda i: -R[i])[:n_mus]
    Sg = np.linalg.pinv(S[good]).T
    names = [MS.MUS[i] for i in good]
    phi0 = [base[MS.MUS[i]][0] for i in good]
    return Sg, names, phi0


def fly(src="true", dec=None, T=3.0, pert_seed=None, shuffle=None, seed=0):
    E = OH.env()
    m, Q0, ZT_W, aid = E["m"], E["Q0"], E["ZT_W"], E["aid"]
    use_net = src in ("circuit", "shuffle")
    if use_net:
        Sg, names, phi0 = dec
        kw = dict(LC.SETUP_KW)
        if shuffle:
            kw["shuffle"] = shuffle
            kw["shuffle_seed"] = seed
        net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
        idx_of = {j: st_idx[j] for j in names if j in st_idx}
        zc = {j: 0j for j in names}
        prev = 0
        shift_prev = np.zeros(len(pref))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        d.qvel[3:6] = np.random.default_rng(pert_seed).normal(0, PERT_SIG, 3)
    mujoco.mj_forward(m, d)
    dtp = m.opt.timestep
    kk = max(P0["sharp"], 1e-3)
    R = np.zeros(9)
    per = 1.0 / P0["freq"]
    NBOX = max(int(per / CF.DT_N), 1)
    ombuf = np.zeros((NBOX, 3))
    ombi = 0
    omsum = np.zeros(3)
    u_cmd = np.array(U_TRIM, float)
    u = np.array(U_TRIM, float)
    om_est = np.zeros(3)
    t_next = 0.0
    ups, zs, alive = [], [], 0
    n_sub = max(int(round(CF.DT_N / dtp)), 1)
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        omsum += d.qvel[3:6] - ombuf[ombi]
        ombuf[ombi] = d.qvel[3:6].copy()
        ombi = (ombi + 1) % NBOX
        o_slow = omsum / NBOX          # ハルテアの周波数選択的復調
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
        if tn >= t_next:                # 1羽ばたき1回だけ指令更新
            t_next += per
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
            v = d.qvel[:3]
            if src == "true":
                ow = o_slow
            elif src == "zero":
                ow = np.zeros(3)
            else:
                ow = om_est
            x = np.array([(12.0 - d.qpos[2]) / 5.0, -v[2] / 30.0,
                          e_b[0], e_b[1],
                          ow[0] / 20.0, ow[1] / 20.0, ow[2] / 20.0])
            u_cmd = np.clip(U_TRIM + np.tanh(K_POL @ x + B_POL) * 0.35,
                            -0.55, 0.55)
        for _ in range(n_sub):
            t = d.time
            u += dtp * (u_cmd - u) / TAU_TW      # 筋単収縮
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
    what = sys.argv[1] if len(sys.argv) > 1 else "true"
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dec = None
    if what in ("circuit", "shuffle"):
        dec = calibrate(shuffle="all" if what == "shuffle" else None, seed=sd)
        print(f"較正完了: {len(dec[1])}筋 (翅{P0['freq']:.1f}Hz)", flush=True)
    s, up, ze = fly(src=what, dec=dec, pert_seed=sd,
                    shuffle="all" if what == "shuffle" else None, seed=sd)
    print(f"{what}(seed{sd}): 生存{s:.2f}s 直立{up:+.2f} 高度誤差{ze:.1f}",
          flush=True)
