#!/usr/bin/env python
"""位相コーディング版ハルテア反射 (0.1ms連成)。

実物の実装に忠実な3点:
  1. ハルテア求心性は羽ばたき毎に好みの位相で〜1発発火(von Mises レート)。
     体回転 ω は発火位相のシフトとして符号化される(コリオリのタイミング符号化)
  2. 反射弧はサブ回路 1,556ニューロン(ハルテア→中継→操舵MN、b1へは単シナプス787syn)
  3. b1 MN のスパイク「位相」をデコード: 位相前進 → 翅振幅増大 (Tu & Dickinson)
     レート化・低域通過を全廃し、周期単位で翅を変調(遅延 ≤1羽ばたき)

mode=prc  : 神経のみ。ω を与えて b1 発火位相がシフトするか (位相応答曲線)
mode=fly  : flybody連成。姿勢のみES + 位相反射 の安定化を検証
"""
import sys
import numpy as np
import pandas as pd
import mujoco
from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
                    Network, mV, ms, Hz, prefs)

import vnc_model

prefs.codegen.target = "numpy"

WBF = 218.0
CYC = 1.0 / WBF
DT_N = 0.0001            # 神経・交換 0.1ms
KAPPA = 25.0             # 位相チューニングの鋭さ
R_PEAK = 4000.0          # von Mises ピークレート (〜1発/周期)
C_PHASE = 0.004          # ω→位相シフト [cycle/(rad/s)]
C_GAIN = 0.01            # ω→ゲイン変調
STEER = ["b1", "b2", "b3", "i1", "i2", "iii1", "iii3", "hg1", "hg2", "hg3", "hg4"]


def build_subnet():
    ids_all, pre, post, w = vnc_model.build_arrays()
    sub = np.load("outputs/subcircuit_ids.npy")
    subset = set(int(s) for s in sub)
    keep = np.array([i for i, b in enumerate(ids_all) if int(b) in subset])
    remap = {old: new for new, old in enumerate(keep)}
    ids = ids_all[keep]
    emask = np.isin(pre, keep) & np.isin(post, keep)
    pre2 = np.array([remap[p] for p in pre[emask]])
    post2 = np.array([remap[p] for p in post[emask]])
    w2 = w[emask]
    idx = {int(b): i for i, b in enumerate(ids)}
    n = len(ids)
    P = vnc_model.PARAMS
    ns = dict(v_0=P["v_0"]*mV, t_mbr=P["t_mbr"]*ms, tau=P["tau"]*ms,
              v_th=P["v_th"]*mV, v_rst=P["v_rst"]*mV)
    eqs = """
    dv/dt = (v_0 - v + g + itn) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    rfc : second
    itn : volt
    """
    neu = NeuronGroup(n, eqs, threshold="v > v_th",
                      reset="v = v_rst; g = 0*mV", refractory="rfc",
                      method="linear", namespace=ns, name="sub")
    neu.v = P["v_0"]*mV; neu.g = 0*mV
    neu.rfc = P["t_rfc"]*ms; neu.itn = 0*mV
    syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                   delay=P["t_dly"]*ms, name="ssyn")
    syn.connect(i=pre2, j=post2)
    syn.w = w2 * P["w_syn"] * mV
    return neu, syn, ids, idx


def setup(gyro=True):
    mns = pd.read_csv("../vnc-connectome/downloads/elife-96084-supp3-v1.csv",
                      encoding="latin1")
    wm = mns[mns.subclass == "wm"]
    steer = wm[wm.target.isin(STEER)][["bodyid", "target"]].copy()
    props = pd.read_feather("neuron-properties.feather")
    som = props.set_index("bodyId")["somaSide"]
    steer["side"] = steer.bodyid.map(som).astype(str).str[:1]
    power = wm[wm.target.astype(str).str.startswith(("DLM", "DVM"))].bodyid
    hal = props[(props["class"] == "sensory neuron")
                & (props.modality == "proprioceptive")
                & props.entryNerve.isin(["DMetaN_L", "DMetaN_R"])]
    neu, syn, ids, idx = build_subnet()
    # 動力筋: 緊張性 / 操舵MN: 高興奮性 (入力シナプスを4倍)
    for b in power:
        if int(b) in idx:
            neu.itn[idx[int(b)]] = 8.5*mV
    st_idx = {}
    for _, r in steer.iterrows():
        if int(r.bodyid) in idx:
            st_idx[f"{r.target}_{r.side}"] = idx[int(r.bodyid)]
    fac = np.ones(len(ids))
    for k, i in st_idx.items():
        fac[i] = 4.0
    syn.w = syn.w[:] * fac[np.array(syn.j[:], dtype=int)]
    # ハルテア: 位相コーディング PoissonGroup (0.1msでレート更新)
    hal_L = [int(b) for b in hal[hal.entryNerve == "DMetaN_L"].bodyId if int(b) in idx]
    hal_R = [int(b) for b in hal[hal.entryNerve == "DMetaN_R"].bodyId if int(b) in idx]
    h_all = hal_L + hal_R
    h_tgt = np.array([idx[b] for b in h_all])
    n_h = len(h_all)
    rng = np.random.default_rng(0)
    pref = rng.uniform(0, 1, n_h)          # 好み位相 [cycle]
    side = np.array([+1]*len(hal_L) + [-1]*len(hal_R))
    pg = PoissonGroup(n_h, rates=0*Hz, name="hal_pg")
    sh = Synapses(pg, neu, on_pre="v_post += %f*mV" %
                  (vnc_model.PARAMS["w_syn"]*vnc_model.PARAMS["f_poi"]),
                  name="hsyn")
    sh.connect(i=np.arange(n_h), j=h_tgt)
    for i in h_tgt:
        neu.rfc[i] = 0*ms
    mon = SpikeMonitor(neu, record=True)
    net = Network(neu, syn, mon, pg, sh)
    return net, mon, pg, pref, side, st_idx, len(ids)


def hal_rates(t, omega, pref, side):
    """位相コーディング: 好み位相±ωシフトの von Mises レート"""
    phc = np.mod(t * WBF, 1.0)
    shift = C_PHASE * (side * omega[0] + omega[1])      # roll:左右逆, pitch:同相
    gain = 1.0 + C_GAIN * (side * omega[0])
    dphi = 2*np.pi*(phc - pref - shift)
    return np.clip(R_PEAK * np.exp(KAPPA*(np.cos(dphi)-1.0)) * gain, 0, 8000)


def mode_prc():
    from brian2 import ms as _ms
    net, mon, pg, pref, side, st_idx, n = setup()
    print(f"subnet {n} neurons")
    results = {}
    for om_r in [0.0, 20.0, -20.0]:
        t0 = float(mon.t[-1]/_ms)/1000.0 if len(mon.t) else 0.0
        T = 0.5
        steps = int(T/DT_N)
        base_t = t0
        for k in range(steps):
            t = base_t + k*DT_N
            pg.rates = hal_rates(t, (om_r, 0, 0), pref, side) * Hz
            net.run(DT_N*1000*_ms)
        trains = mon.spike_trains()
        for mu in ["b1_L", "b1_R"]:
            if mu not in st_idx: continue
            st = np.array(trains[st_idx[mu]]/_ms)/1000.0
            st = st[st > base_t + 0.1]
            phases = np.mod(st*WBF, 1.0)
            if len(phases) > 3:
                mean_ph = np.angle(np.mean(np.exp(2j*np.pi*phases)))/(2*np.pi) % 1.0
                Rv = np.abs(np.mean(np.exp(2j*np.pi*phases)))
                results[(om_r, mu)] = (len(st)/ (T-0.1), mean_ph, Rv)
    print("\n=== b1 位相応答 (発火率, 平均位相, 位相固定度R) ===")
    for (om, mu), (r, mph, Rv) in sorted(results.items()):
        print(f"ω_roll={om:+5.0f}: {mu}  {r:6.1f}Hz  位相{mph:.3f}  R={Rv:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prc":
        mode_prc()


def mode_fly():
    """0.1ms連成: 姿勢のみES + b1位相デコード反射の飛行テスト"""
    from brian2 import ms as _ms
    import os
    sys.path.insert(0, "../fly-flight-sim")
    import fly_flight2 as FF
    ckp = np.load("../fly-flight-sim/outputs/hover_policy.npz")
    theta = ckp["theta"]; N_X, N_U = 7, 5
    U_SCALE = np.array([0.5, 0.5, 0.4, 0.4, 0.3])
    K_att = theta[:N_U*N_X].reshape(N_U, N_X).copy(); K_att[:, 4:7] = 0.0
    b_pol = theta[N_U*N_X:]
    Pw = {k: float(v) for k, v in dict(np.load(
        "../fly-flight-sim/outputs/hover_params.npz")).items()}
    TH = np.deg2rad(47.5); Q0 = [np.cos(-TH/2), 0, np.sin(-TH/2), 0]
    _R0 = np.zeros(9); mujoco.mju_quat2Mat(_R0, np.array(Q0))
    ZT_W = np.array([_R0[2], _R0[5], _R0[8]])

    def trial(G_reflex, T=2.0, warm_T=0.35):
        net, mon, pg, pref, side, st_idx, n = setup()
        i_b1L = st_idx.get("b1_L"); i_b1R = st_idx.get("b1_R")
        cwd = os.getcwd(); os.chdir("../fly-flight-sim")
        try: m = FF.build_model()
        finally: os.chdir(cwd)
        d = mujoco.MjData(m); dtp = m.opt.timestep
        aid = {nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
               for nm in ["wing_yaw_left","wing_roll_left","wing_pitch_left",
                          "wing_yaw_right","wing_roll_right","wing_pitch_right"]}
        mujoco.mj_resetData(m, d); d.qpos[2] = 12.0; d.qpos[3:7] = Q0
        mujoco.mj_forward(m, d)
        R = np.zeros(9)
        # b1位相の追跡 (基準位相はPRC実験の実測値を定数で使用)
        last_ph = {"L": None, "R": None}
        PH0 = {"L": 0.210, "R": 0.954}
        dphi = {"L": 0.0, "R": 0.0}
        prev_nsp = 0
        ups, n_alive = [], 0
        n_steps = int(T/DT_N)
        for k in range(n_steps):
            t = k*DT_N
            om = d.qvel[3:6]
            pg.rates = hal_rates(t, (om[0], om[1], 0), pref, side) * Hz
            net.run(DT_N*1000*_ms)
            # 新スパイクの位相を読む (b1のみ)
            nsp = mon.num_spikes
            if nsp > prev_nsp:
                ii = np.array(mon.i[prev_nsp:nsp])
                tt_s = np.array(mon.t[prev_nsp:nsp]/_ms)/1000.0
                for iN, tS in zip(ii, tt_s):
                    for sd, iB in (("L", i_b1L), ("R", i_b1R)):
                        if iB is not None and iN == iB:
                            last_ph[sd] = float(np.mod(tS*WBF, 1.0))
                prev_nsp = nsp
            if t >= 0.12:
                for sd in ("L","R"):
                    if last_ph[sd] is not None:
                        dv = (last_ph[sd] - PH0[sd] + 0.5) % 1.0 - 0.5
                        dphi[sd] = 0.9*dphi[sd] + 0.1*dv     # τ≈1msの平滑
            ampL = np.clip(1.0 + G_reflex*(-dphi["L"]), 0.7, 1.4)   # 位相前進(負dv)→振幅増
            ampR = np.clip(1.0 + G_reflex*(-dphi["R"]), 0.7, 1.4)
            # 姿勢のみESポリシー (物理ステップ毎)
            for kp in range(int(DT_N/dtp)):
                tt = t + kp*dtp
                mujoco.mju_quat2Mat(R, d.qpos[3:7])
                zc = np.array([R[2],R[5],R[8]])
                e_b = R.reshape(3,3).T @ np.cross(zc, ZT_W)
                omp = d.qvel[3:6]
                x = np.array([(12.0-d.qpos[2])/5.0, -d.qvel[2]/30.0, e_b[0], e_b[1],
                              omp[0]/20.0, omp[1]/20.0, omp[2]/20.0])
                u = np.tanh(K_att @ x + b_pol) * U_SCALE
                amp = np.clip(1.0 + u[0], 0.5, 1.6)
                env0 = min(tt/0.03, 1.0)
                ph2 = 2*np.pi*Pw["freq"]*tt
                s = np.sin(ph2); kk = max(Pw["sharp"],1e-3)
                rot = np.tanh(kk*np.cos(ph2+Pw["phase"]))/np.tanh(kk)
                eL = env0*amp*ampL; eR = env0*amp*ampR
                d.ctrl[:] = 0
                d.ctrl[aid["wing_yaw_left"]] = eL*(Pw["yaw_amp"]*s + u[1] + u[2])
                d.ctrl[aid["wing_yaw_right"]] = eR*(Pw["yaw_amp"]*s + u[1] - u[2])
                d.ctrl[aid["wing_pitch_left"]] = eL*(-Pw["pitch_amp"]*rot+Pw["pitch_bias"]+u[3]+u[4])
                d.ctrl[aid["wing_pitch_right"]] = eR*(-Pw["pitch_amp"]*rot+Pw["pitch_bias"]+u[3]-u[4])
                d.ctrl[aid["wing_roll_left"]] = eL*Pw["roll_amp"]*np.sin(2*ph2)
                d.ctrl[aid["wing_roll_right"]] = eR*Pw["roll_amp"]*np.sin(2*ph2)
                mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos[2]) or d.qpos[2] < 0.5:
                break
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            ups.append(np.array([R[2],R[5],R[8]]) @ ZT_W)
            n_alive = k+1
        return n_alive*DT_N, float(np.mean(ups)) if ups else 0

    import itertools
    global C_PHASE
    for g, cp in [(6.0, 0.004), (6.0, 0.008), (12.0, 0.008), (20.0, 0.012)]:
        C_PHASE = cp
        s, up = trial(g)
        print(f"G={g} C_PHASE={cp}: 生存{s:.2f}s 直立度{up:.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prc":
        mode_prc()
    elif len(sys.argv) > 1 and sys.argv[1] == "fly":
        mode_fly()
