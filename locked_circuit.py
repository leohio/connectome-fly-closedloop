#!/usr/bin/env python
"""統合28: locked透過段 (決定論的位相発火求心性 + 電気シナプスモデル) の
実回路張力基底を採取し、教師波形へLSQ再適合して飛行検証する。

透過段モデルの宣言 (ハリボテ回避のための正直な記載):
- 求心性の1周期1発・位相固定発火 = 桿状感覚子の実測生理 (Yarger & Fox 2018)
- pref反転2クラスタ = ストローク反転での発火集中 (同上)
- 電気シナプス8mV/0.5ms = ハルテア→b1電気結合 (Fayyazuddin & Dickinson 1996)。
  化学コネクトーム (MANC) に存在しないため実配線エッジの上にモデルとして追加。
配線 (どの求心性がどのMNへ至るか) は traced-connections の実データのまま。
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF

SETUP_KW = dict(afferent_mode="locked", pref_mode="reversal",
                electrical=True, mn_ahp=True)
NB = 200

# ハルテア-翅の機械的周波数結合 (Deora, Singh & Sane 2015):
# ハルテアは胸部の同一振動系で駆動され、翅と厳密に同周波数で拍動する。
# 求心性の位相基準を実際の翅拍動周波数に一致させる (従来は218Hz固定で
# 翅202.3Hzと16Hzずれ、62msごとに位相が一周スリップしていた)
PR.WBF = float(PR._fly_env()["Pw"]["freq"])

# ハルテア→操舵筋の位相シフト感度を生理値へ。実バエのb1位相シフトは
# 大旋回時でも数%周期 (Tu & Dickinson 1996: サッカード ~35rad/s で
# 0.03周期程度)。従来の0.03 cycle/(rad/s) は30倍過大で、ω=8で0.24周期
# =86°もずれ、隣の求心性volleyへ乗り換える半周期跳びを起こしていた
PR.C_PHASE = 0.0015


TAU_P = 0.010     # 筋活動位相の推定時定数 [s] (1周期1発を2周期分平均)
OMLOG = None      # not None にすると復号ωと真のω(復調後)を記録する


def measure_basis(T=0.8, want_phase=False):
    """ω=0でlocked回路を走らせ、12筋の実張力波形(200bin)を採取。
    want_phase=Trueなら各筋の基準活動位相φ0 (ストローク位相基準) も返す"""
    net, mon, pg, pref, side, st_idx, n = PR.setup(**SETUP_KW)
    ch_idx = {mu: st_idx[mu] for mu in CF.NAMES if mu in st_idx}
    us = {mu: 0.0 for mu in ch_idx}
    fs = {mu: 0.0 for mu in ch_idx}
    acc = {mu: np.zeros(NB) for mu in ch_idx}
    zc = {mu: 0j for mu in ch_idx}
    zsum = {mu: 0j for mu in ch_idx}
    nz = 0
    cnt = np.zeros(NB)
    prev = 0
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        net.run(CF.DT_N * 1000 * _ms)
        ph_w = (tn * PR.WBF) % 1.0
        nsp = mon.num_spikes
        if nsp > prev:
            for iN in np.array(mon.i[prev:nsp]):
                for mu, im in ch_idx.items():
                    if int(iN) == im:
                        us[mu] += 1.0 / CF.TAU_A
                        zc[mu] += np.exp(2j * np.pi * ph_w)
            prev = nsp
        b = min(int(ph_w * NB), NB - 1)
        for mu in ch_idx:
            us[mu] -= CF.DT_N * us[mu] / CF.TAU_A
            fs[mu] += CF.DT_N * (us[mu] - fs[mu]) / CF.TAU_A
            zc[mu] -= CF.DT_N * zc[mu] / TAU_P
            if tn > 0.2:
                acc[mu][b] += fs[mu]
                zsum[mu] += zc[mu]
        if tn > 0.2:
            cnt[b] += 1
            nz += 1
    Bc = {mu: acc[mu] / np.maximum(cnt, 1) for mu in ch_idx}
    if want_phase:
        phi0 = {mu: np.angle(zsum[mu] / max(nz, 1)) / (2 * np.pi)
                for mu in ch_idx}
        return Bc, phi0
    return Bc


def fit(Bc):
    """回路実基底で教師u1-4波形へLSQ適合"""
    M = CF.chan_matrix()
    uref = CF.UREF
    cols, names2, means = [], [], []
    for j, mu in enumerate(CF.NAMES):
        if mu not in Bc:
            continue
        dT = Bc[mu] - Bc[mu].mean()
        cols.append(np.outer(dT, M[j]).reshape(-1))
        names2.append(mu)
        means.append(Bc[mu].mean())
    A = np.stack(cols, axis=1)
    y = uref[:, 1:5].reshape(-1)
    g, *_r = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ g) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return g, names2, np.array(means), r2


def fly(gc, names2, means, mode="ff", T=2.0, axis_gain=None,
        rec_gain=None, phase_gain=None, phi0=None, pert_seed=None,
        decode=None, K=None, shuffle=None, shuffle_seed=0):
    """locked回路で飛行。mode='ff'はω=0、'closed'はハルテア位相シフト帰還
    axis_gain=(cR,cP,cY): 軸別反射ゲイン (生物学対応: ハルテア反射の
    ゲイン調律は発達・飛行経験で較正される)。None時はPR.C_PHASE一律"""
    env = PR._fly_env()
    Pw, Q0, ZT_W = env["Pw"], env["Q0"], env["ZT_W"]
    m = CF.OPT.get_model()
    aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
           for nm in ["wing_yaw_left", "wing_roll_left", "wing_pitch_left",
                      "wing_yaw_right", "wing_roll_right",
                      "wing_pitch_right"]}
    M = CF.chan_matrix()
    jmap = [CF.NAMES.index(mu) for mu in names2]
    kw = dict(SETUP_KW)
    if rec_gain is not None:
        kw["recruit"] = True
    if shuffle:
        kw["shuffle"] = shuffle
        kw["shuffle_seed"] = shuffle_seed
    # 較正復号モード: 回路の筋位相からωを復号し、ヒンジ有効性行列Bで
    # 望むトルクをチャネル配分する。配線は不変、較正量のみ外部から与える
    dec = None
    if decode is not None:
        Sg, phg, jg, Bpinv = decode
        zc2 = {j: 0j for j in jg}
        dec = True
    net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
    ch_idx = {mu: st_idx[mu] for mu in names2 if mu in st_idx}
    st_idx_all = dict(st_idx) if dec else {}
    us = {mu: 0.0 for mu in names2}
    fs = {mu: 0.0 for mu in names2}
    # ヒンジの位相→振幅変換 (Tu & Dickinson 1996; Lindsay+ 2017):
    # 操舵筋は張力の大小ではなく「ストローク内のいつ収縮するか」で翅を操舵する。
    # 各筋の活動位相を複素EMAで推定し、基準位相からのずれを操舵量に写す。
    zc = {mu: 0j for mu in names2}
    use_ph = phase_gain is not None and phi0 is not None
    prev = 0
    shift_prev = np.zeros(len(pref))
    # ハルテアによる周波数選択的復調 (Nalbach 1993; Fox & Daniel 2008):
    # ハルテアは翅と同周波数・逆位相で拍動し、自身の運動によるコリオリ力を
    # 測る。機体の羽ばたき反動振動 (±600rad/s, 拍動と同周波数) は積により
    # 直流・2f成分となり、体の剛体回転だけが f 成分として現れる。つまり
    # ハルテアは「1拍動周期の移動平均」を遅延ほぼ無しで実現している。
    # 従来の50ms一次ローパスは70msのループ遅延を生み、0.3sで墜ちる不安定系
    # には致命的だった。1周期ボックスカー (遅延=半周期2.5ms) に置き換える
    NBOX = max(int((1.0 / Pw["freq"]) / CF.DT_N), 1)
    ombuf = np.zeros((NBOX, 3))
    ombi = 0
    omsum = np.zeros(3)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 12.0
    d.qpos[3:7] = Q0
    if pert_seed is not None:
        # 初期外乱 (突風相当): 姿勢制御の頑健性を測るための再現可能な摂動。
        # 単発試行はカオス的で分散が大きいため、複数seedの平均で評価する
        rg = np.random.default_rng(pert_seed)
        d.qvel[3:6] = rg.normal(0, 1.5, 3)
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    dtp = m.opt.timestep
    kk = max(Pw["sharp"], 1e-3)
    alive, ups = 0, []
    for kn in range(int(T / CF.DT_N)):
        tn = kn * CF.DT_N
        om = d.qvel[3:6]
        if mode == "closed":
            omsum += om - ombuf[ombi]
            ombuf[ombi] = om
            ombi = (ombi + 1) % NBOX
            o = np.clip(omsum / NBOX, -12, 12)
            if axis_gain is None:
                cR = cP = cY = PR.C_PHASE
            else:
                cR, cP, cY = axis_gain
            shift = (cR * side * o[0] + cP * o[1]
                     + cY * side * np.cos(2 * np.pi * pref) * o[2])
            pg.v = pg.v - (shift - shift_prev)
            shift_prev = shift
            if rec_gain is not None:
                from brian2 import Hz as _Hz
                kR, kP, kY = rec_gain
                drv = (kR * side * o[0] + kP * o[1]
                       + kY * side * np.cos(2 * np.pi * pref) * o[2])
                PR.REC_PG.rates = np.clip(drv, 0, 600) * _Hz
        net.run(CF.DT_N * 1000 * _ms)
        ph_w = (tn * Pw["freq"]) % 1.0
        nsp = mon.num_spikes
        if nsp > prev:
            ev = np.exp(2j * np.pi * ph_w)
            for iN in np.array(mon.i[prev:nsp]):
                for mu, im in ch_idx.items():
                    if int(iN) == im:
                        us[mu] += 1.0 / CF.TAU_A
                        if use_ph:
                            zc[mu] += ev
                if dec:
                    for j in jg:
                        if int(iN) == st_idx_all.get(j, -1):
                            zc2[j] += ev
            prev = nsp
        u14 = np.zeros(4)
        for jj, mu in enumerate(names2):
            us[mu] -= CF.DT_N * us[mu] / CF.TAU_A
            fs[mu] += CF.DT_N * (us[mu] - fs[mu]) / CF.TAU_A
            u14 += gc[jj] * (fs[mu] - means[jj]) * M[jmap[jj]]
            if use_ph:
                zc[mu] -= CF.DT_N * zc[mu] / TAU_P
                if abs(zc[mu]) > 1e-3:
                    dphi = np.angle(zc[mu]) / (2 * np.pi) - phi0[jj]
                    dphi = (dphi + 0.5) % 1.0 - 0.5   # [-0.5, 0.5)へ巻き戻し
                    u14 += phase_gain[jj] * dphi * M[jmap[jj]]
        if dec:
            # 回路の筋活動位相からωを復号 → 望む復元トルク → チャネル配分
            dphi = np.zeros(len(jg))
            okd = True
            for q, j in enumerate(jg):
                zc2[j] -= CF.DT_N * zc2[j] / TAU_P
                if abs(zc2[j]) < 1e-3:
                    okd = False
                    break
                dd = np.angle(zc2[j]) / (2 * np.pi) - phg[q]
                dphi[q] = (dd + 0.5) % 1.0 - 0.5
            if okd and tn > 0.15:
                om_est = dphi @ Sg
                a_des = -np.asarray(K, float) * om_est
                u14 = u14 + a_des @ Bpinv
                if OMLOG is not None and kn % 20 == 0:
                    OMLOG.append((tn, *om_est, *o))
        if tn < 0.2:   # 起動橋渡し (回路の張力立ち上がり待ち)
            phc = (tn * Pw["freq"]) % 1.0
            i = min(int(phc * NB), NB - 1)
            w = tn / 0.2
            u14 = (1 - w) * CF.Z["recon"][i] + w * u14
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
                                                  + Pw["pitch_bias"]
                                                  + u[3] + u[4])
            d.ctrl[aid["wing_pitch_right"]] = e * (-Pw["pitch_amp"] * rot
                                                   + Pw["pitch_bias"]
                                                   + u[3] - u[4])
            d.ctrl[aid["wing_roll_left"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            d.ctrl[aid["wing_roll_right"]] = e * Pw["roll_amp"] * np.sin(2 * ph2)
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
            break
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        ups.append(np.array([R[2], R[5], R[8]]) @ ZT_W)
        alive = kn + 1
    return alive * CF.DT_N, float(np.mean(ups)) if ups else 0


if __name__ == "__main__":
    Bc, phi0d = measure_basis(want_phase=True)
    for mu, w in Bc.items():
        print(f"  {mu}: p-p={w.max()-w.min():.1f} mean={w.mean():.1f}",
              flush=True)
    gc, names2, means, r2 = fit(Bc)
    print(f"locked回路実基底LSQ: {len(names2)}筋 R²={r2:.3f}", flush=True)
    np.savez("outputs/locked_fit.npz", g=gc, names=np.array(names2),
             means=means, basis=np.stack([Bc[mu] for mu in names2]), r2=r2,
             phi0=np.array([phi0d[mu] for mu in names2]))
    s, up = fly(gc, names2, means, mode="ff")
    print(f"locked回路FF(ω=0)      生存{s:.2f}s 直立度{up:+.2f}", flush=True)
    s, up = fly(gc, names2, means, mode="closed")
    print(f"locked回路閉ループ        生存{s:.2f}s 直立度{up:+.2f}", flush=True)
