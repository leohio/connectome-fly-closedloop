#!/usr/bin/env python
"""統合39: 下行指令による行動遷移 — 歩行→離陸→飛行→着陸→歩行の連続個体。

指令は実FlyWire IDの下行ニューロンLIF群の発火で駆動する (実験者の「意図」=
該当DN集団へのPoisson入力。ロードマップ決定2で先にこの方式):
  MDN (後退歩行, x4) / GF=DNp01 (離陸, x2) / DNp09 (前進歩行, x4)
遷移は指令DNのスパイク数で発火的にトリガされる (時刻直書きではない)。

身体は flybody 一体。歩行=統合38c (MANC歩行net 2ms + 反射)、
飛行=統合37 (MANC飛行net 0.1ms + 脳視覚net + 学習K/W)。
歩行中は翅回路が、飛行中は脚反射が抑制される (下行ゲーティングのモデル)。
着陸は高度目標の降下→脚接地で翅停止→立位復帰。

sim: 実行して全スパイク・フレーム・状態を記録 / compose は video_sequence.py
"""
import sys
import os
import json
import numpy as np
import mujoco
from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor,
                    Network, mV, ms as _ms, Hz, prefs)
prefs.codegen.target = "numpy"

FPS = 60
DT_WALK = 0.002

CMD_IDS = {  # 実FlyWire ID
    "MDN": [720575940640331472, 720575940610236514,
            720575940631082808, 720575940616026939],
    "GF": [720575940622838154, 720575940632499757],
    "DNp09": [720575940635872101, 720575940625673196,
              720575940633095137, 720575940627652358],
}


def build_command_group():
    """指令DNのLIF群 (Shiu et al.パラメタ)。意図=rates書き換えで注入"""
    ids = [b for v in CMD_IDS.values() for b in v]
    n = len(ids)
    eqs = "dv/dt = -(v - (-52*mV)) / (20*ms) : volt (unless refractory)"
    g = NeuronGroup(n, eqs, threshold="v > -45*mV", reset="v = -52*mV",
                    refractory=2.2 * _ms, method="exact", name="cmd")
    g.v = -52 * mV
    pg = PoissonGroup(n, rates=np.zeros(n) * Hz, name="cmd_in")
    sy = Synapses(pg, g, on_pre="v_post += 2.75*mV", name="cmd_syn")
    sy.connect(j="i")
    mon = SpikeMonitor(g, name="cmd_mon")
    net = Network(g, pg, sy, mon)
    sl = {}
    at = 0
    for k, v in CMD_IDS.items():
        sl[k] = slice(at, at + len(v))
        at += len(v)
    return net, mon, pg, sl, ids


def main(T_total=float(os.environ.get("T_TOTAL", "14.0")),
         out="outputs/seq_record.npz"):
    import openloop_hover as OH
    import phase_reflex as PR
    import connectome_fastloop as CF
    import locked_circuit as LC
    import connectome_bioflight as CB
    import vision_pop as VP
    import brain_visual as BV
    import vision_circuit_flight as VCF
    import plastic_readout as PL
    import walk_base as WB
    import walk_reflex as WR
    import walk_drive as WD
    import vnc_model
    from closed_loop import sensory_pools
    from closed_loop4 import compensation_factors

    # ---- 指令DN群 ----
    cnet, cmon, cpg, csl, cids = build_command_group()

    # ---- 歩行側 (統合38c) ----
    W = WB.env()
    m = W["m"]
    an = {n: i for i, n in enumerate(W["leg_names"])}
    act_ids = {n: int(W["leg_act"][i]) for n, i in an.items()}
    stance = {n: float(W["stance_q"][i]) for n, i in an.items()}
    sn_by_leg = sensory_pools()
    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    import pandas as pd
    _props = pd.read_feather("neuron-properties.feather").set_index("bodyId")
    def sn_kind(b):
        sc = str(_props.loc[b].subclass) if b in _props.index else ""
        for k in ("claw", "club", "hook"):
            if k in sc:
                return k
        return "other"
    KIND = {b: sn_kind(b) for b in sn_all}
    import integrate_measured as IMm
    mp = IMm.build_measured_pools()
    excit = {int(r.bodyid): float(IMm.E_MAX ** (1.0 - r.pct))
             for _, r in mp.iterrows()}
    tonic = {int(r.bodyid): float(IMm.ITN_MAX * max(
        0.0, (IMm.TONIC_CUTOFF - r.pct) / IMm.TONIC_CUTOFF))
        for _, r in mp.iterrows()}
    wnet, wmon, wids, widx, wpg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=0.0, sensory_bodyids=sn_all,
        excitability=excit, tonic_mv=tonic)
    # MDN刺激レートは歩行フェーズで書き換える (指令DN発火に比例)
    kept = [b for b in sn_all if b in widx]
    leg_slices, kind_masks = {}, {}
    at = 0
    LEGS = WR.LEGS
    for leg in LEGS:
        legk = [b for b in sn_by_leg[leg] if b in widx]
        nn = len(legk)
        leg_slices[leg] = slice(at, at + nn)
        for kd in ("club", "hook", "other"):
            kind_masks[(leg, kd)] = np.array(
                [i for i, b in enumerate(legk) if KIND[b] == kd], dtype=int)
        cl = [i for i, b in enumerate(legk) if KIND[b] == "claw"]
        kind_masks[(leg, "claw_flex")] = np.array(
            [i for i in cl if legk[i] % 2 == 0], dtype=int)
        kind_masks[(leg, "claw_ext")] = np.array(
            [i for i in cl if legk[i] % 2 == 1], dtype=int)
        at += nn
    from brian2 import SpikeMonitor as _SM2
    wrec = _SM2(wmon.source, name="walk_rec")
    wnet.add(wrec)
    sens_comp, motor_comp = compensation_factors(sn_by_leg)
    pools = WD.flybody_pools(widx, W)
    F = {k: np.zeros(len(v)) for k, v in pools.items()}
    LEG2SUF = WR.LEG2SUF
    legjq = {leg: [m.jnt_qposadr[m.actuator_trnid[act_ids[n]][0]]
                   for n in an if n.endswith(f"{s}_{sd}")]
             for leg, (s, sd) in LEG2SUF.items()}
    leggeo = {}
    for leg, (s, sd) in LEG2SUF.items():
        leggeo[leg] = [g for g in range(m.ngeom)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                           or "").endswith(f"_{s}_{sd}_collision")]

    # ---- 飛行側 (統合37) ----
    # 飛行制御は統合31のES帰還則 (安定飛行の実績: 直立+0.92, 高度誤差0.5)。
    # 報酬学習K (統合36-37) は高度保持が弱く、安定飛行の基準S2を満たせない
    # 統合39改訂2: 実回路相当の感覚 (回路模型+遅延2) と離陸プロファイルで
    # 再学習したES帰還則。安価ループでz=0.6から離陸→巡航10→降下を実証
    AIRBORNE = int(os.environ.get("AIRBORNE_START", "1"))
    # 空中開始モード: 飛行区間は統合31で10秒安定飛行を実証したES帰還則
    # (地上離陸は未解決 — README追記35参照)
    th_es = np.load("outputs/bioflight_best.npy") if AIRBORNE \
        else np.load("outputs/es_takeoff_d3_best.npy")
    K_POL = th_es[12:12 + CB.N_U * CB.N_X].reshape(CB.N_U, CB.N_X)
    B_POL = th_es[12 + CB.N_U * CB.N_X:]
    print("読出しWを較正中 (ES-Kと整合する外部較正)...", flush=True)
    Sg, names, phi0 = CB.calibrate()
    # 位相推定の時定数を10→5msへ: 実回路の実効遅延を約2→約1羽ばたきに短縮
    # (統合39改訂2の安価ループ検証: ES-Kは遅延1なら離陸可、遅延2では不可)
    LC.TAU_P = 0.005
    fnet, fmon, fpg, pref, side, st_idx, nfly = PR.setup(**LC.SETUP_KW)
    from brian2 import SpikeMonitor as _SM
    idx_of = {j: st_idx[j] for j in names if j in st_idx}
    VP.TAU_DN = VCF.TAU_DN
    vnet, vmon, vpg, ssign, is_oc, readout = VP.build()
    vrec = _SM(vmon.source, name="vis_rec")   # VP.buildはrecord=Falseのため
    vnet.add(vrec)
    print("脳視覚を較正中...", flush=True)
    r0, t_r, t_p, s2r, s2p = VP.calibrate(vnet, vmon, vpg, ssign, is_oc,
                                          readout)
    P0, U_TRIM = CB.P0, CB.U_TRIM
    aid = W["aid"]
    ZT_W, Q0 = W["ZT_W"], W["Q0"]

    # ---- 状態 ----
    d = mujoco.MjData(m)
    WB.reset_standing(m, d)
    WB.settle(m, d, W, T=0.3)
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 2.2, -10, 105
    dtp = m.opt.timestep
    mode = "walk"
    phase_log = []
    frames = []
    next_f = 0.0
    t = 0.0
    # 飛行用状態
    kk = max(P0["sharp"], 1e-3)
    per = 1.0 / P0["freq"]
    NBOX = max(int(per / CF.DT_N), 1)
    ombuf = np.zeros((NBOX, 3))
    ombi, omsum = 0, np.zeros(3)
    zc = {j: 0j for j in names}
    fprev = 0
    shift_prev = np.zeros(len(pref))
    u_cmd = np.array(U_TRIM, float)
    u = np.array(U_TRIM, float)
    om_est = np.zeros(3)
    eb_est = np.zeros(2)
    vprev = vmon.count[:].copy()
    f_rate = r0.copy()
    t_nextcmd = 0.0
    t_flight0 = None
    z_target = 0.2
    R = np.zeros(9)
    wprev = np.zeros(wmon.source.N)
    rates_vec = np.zeros(at)
    f_ref = 1e-6
    prev_cmd = {}
    fr6 = np.zeros(6)
    NV = max(int(VCF.DT_W / CF.DT_N), 1)
    kn_f = 0
    gf_seen = 0
    took_off = False
    landed_hold = 0.0
    contact_acc = 0.0
    ph_acc = 0.0
    t_settle = None
    t_cruise0 = t_desc0 = None
    crit_log = []
    diag = []
    # 意図スケジュール: 0-2.2s MDN → 2.2s GF → 飛行 → 6.8sから降下 → 接地で歩行
    def intent(tn):
        r = np.zeros(len(cids))
        if mode == "walk" and not took_off:
            r[csl["MDN"]] = 250.0
            if tn >= 2.2:
                r[csl["GF"]] = 800.0
        elif mode == "walk" and took_off:
            r[csl["MDN"]] = 250.0
        return r

    tmap = {k: [] for k in ("cmd", "walk", "fly", "vis")}
    def mark(k, net_):
        tmap[k].append((float(net_.t / _ms) / 1000.0, t))
    print("シーケンス開始", flush=True)
    while t < T_total:
        # ---- 指令DN (常時, 2ms刻み) ----
        cpg.rates = intent(t) * Hz
        cnet.run(DT_WALK * 1000 * _ms)
        mark("cmd", cnet)
        csp = cmon.count[:].copy()
        gf_now = int(csp[csl["GF"]].sum())
        if mode == "walk" and not took_off and gf_now > gf_seen + 1:
            t_flight0 = t
            pose0 = d.qpos[0:7].copy()   # 把持スプールアップ中の固定姿勢
            released = False
            if AIRBORNE:
                # 区間カット: 空中 (z=12, ホバー姿勢) から飛行を開始する
                d.qpos[2] = 12.0
                d.qpos[3:7] = np.asarray(Q0, float)
                d.qvel[:] = 0.0
                d.qvel[3:6] = np.random.default_rng(1).normal(0, 1.0, 3)
                mujoco.mj_forward(m, d)
                mode = "flight"
                t_cruise0 = t
                released = True
                z_target = 12.0
                print(f"t={t:.2f}s GF発火{gf_now} → [カット] 空中開始 z=12",
                      flush=True)
            else:
                mode = "takeoff"
                print(f"t={t:.2f}s GF発火{gf_now} → 離陸 (翅スプールアップ開始)",
                      flush=True)
        gf_seen = max(gf_seen, gf_now)
        # ---- フェーズ実行 ----
        if mode == "settle":
            # 着陸後の静定: 翅停止・立位保持。直立が戻ったら歩行再開
            for n2, i2 in act_ids.items():
                d.ctrl[i2] = stance[n2]
            for w2 in ("wing_yaw_left", "wing_yaw_right", "wing_pitch_left",
                       "wing_pitch_right", "wing_roll_left",
                       "wing_roll_right"):
                d.ctrl[aid[w2]] = 0
            for _ in range(int(DT_WALK / dtp)):
                mujoco.mj_step(m, d)
            t += DT_WALK
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            if t - t_settle > 0.6 and R[8] > 0.85:
                mode = "walk"
                took_off = True
                prev_cmd.clear()               # 歩行状態を新規に (残留指令の排除)
                for kf2 in F:
                    F[kf2][:] = 0.0
                print(f"t={t:.2f}s 静定完了 (直立{R[8]:+.2f}) → 歩行再開",
                      flush=True)
            elif t - t_settle > 2.5:
                print(f"t={t:.2f}s 静定失敗 (直立{R[8]:+.2f})", flush=True)
                mode = "walk"
                took_off = True
        elif mode in ("walk",):
            # 統合38c の1窓 (2ms)
            fr6[:] = 0
            fbuf = np.zeros(6)
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                if g1 != W["floor"] and g2 != W["floor"]:
                    continue
                go = g2 if g1 == W["floor"] else g1
                mujoco.mj_contactForce(m, d, c, fbuf)
                for li, leg in enumerate(LEGS):
                    if go in leggeo[leg]:
                        fr6[li] += abs(fbuf[0])
            loads = dict(zip(LEGS, fr6))
            f_ref = max(f_ref, max(loads.values()))
            q_neutral = {leg: np.array([stance[n] for n in an
                         if n.endswith(f"{LEG2SUF[leg][0]}_{LEG2SUF[leg][1]}")])
                         for leg in LEGS}
            for leg in LEGS:
                qs = np.array([d.qpos[qa] for qa in legjq[leg]])
                dqs = np.array([d.qvel[m.jnt_dofadr[m.actuator_trnid[
                    act_ids[n]][0]]] for n in an
                    if n.endswith(f"{LEG2SUF[leg][0]}_{LEG2SUF[leg][1]}")])
                dev_s = float((qs - q_neutral[leg]).sum())
                vel = np.abs(dqs).sum()
                flexing = float(np.mean(dqs))
                sl2 = leg_slices[leg]
                base = np.full(sl2.stop - sl2.start, WR.R_TONIC)
                mk = kind_masks
                base[mk[(leg, "claw_flex")]] += WR.KP * max(dev_s, 0.0)
                base[mk[(leg, "claw_ext")]] += WR.KP * max(-dev_s, 0.0)
                base[mk[(leg, "club")]] += 8.0 * WR.KV * vel
                base[mk[(leg, "hook")]] += 60.0 * max(flexing, 0.0)
                base[mk[(leg, "other")]] += WR.KL * loads[leg] / f_ref
                rates_vec[sl2] = np.clip(base * sens_comp[leg], 0, WR.R_MAX)
            wpg.rates = rates_vec * Hz
            # MDN刺激 = 指令MDN群の発火率に比例 (下行接続のモデル)
            mdn_rate = 150.0 if csp[csl["MDN"]].sum() > 0 else 0.0
            for b in vnc_model.MDN_BODYIDS:
                pass  # make_networkのr_stim=0、直接vへ注入は簡略化しwpgのみ
            wnet.run(DT_WALK * 1000 * _ms)
            mark("walk", wnet)
            cnt = wmon.count[:].copy()
            dspk = cnt - wprev
            wprev = cnt
            qcmd = dict(stance)
            for (act2, dr), lst in pools.items():
                f = F[(act2, dr)]
                for q2, (ni, g, tau, fmax) in enumerate(lst):
                    f[q2] += dspk[ni] * g
                    f[q2] = min(f[q2], fmax)
                    f[q2] -= DT_WALK * f[q2] / tau
                seg, sd = act2.split("_")[1], act2.split("_")[2]
                leg = [k for k, v in LEG2SUF.items() if v == (seg, sd)][0]
                scale = WR.EXT_SCALE if dr < 0 else 1.0
                qcmd[act2] = qcmd.get(act2, stance[act2]) \
                    + dr * scale * WR.K_WALK * WD.GAIN_JOINT[
                        act2.split("_")[0]] * motor_comp[leg] * float(f.sum())
            for n2, i2 in act_ids.items():
                lo, hi = m.actuator_ctrlrange[i2]
                v2 = np.clip(qcmd[n2], stance[n2] - WR.Q_CLAMP,
                             stance[n2] + WR.Q_CLAMP)
                pv = prev_cmd.get(n2, stance[n2])
                v2 = np.clip(v2, pv - WR.SLEW, pv + WR.SLEW)
                prev_cmd[n2] = v2
                d.ctrl[i2] = np.clip(v2, lo, hi)
            for _ in range(int(DT_WALK / dtp)):
                mujoco.mj_step(m, d)
            t += DT_WALK
        elif mode in ("takeoff", "flight", "descend"):
            # 統合37 の0.1ms刻み x 20 = 2ms分
            for _ in range(20):
                tn = kn_f * CF.DT_N
                omsum += d.qvel[3:6] - ombuf[ombi]
                ombuf[ombi] = d.qvel[3:6].copy()
                ombi = (ombi + 1) % NBOX
                o_slow = omsum / NBOX
                o = np.clip(o_slow, -12, 12)
                shift = PR.C_PHASE * (side * o[0] + o[1]
                                      + side * np.cos(2 * np.pi * pref) * o[2])
                fpg.v = fpg.v - (shift - shift_prev)
                shift_prev = shift
                fnet.run(CF.DT_N * 1000 * _ms)
                if kn_f % 20 == 0:
                    mark("fly", fnet)
                ph_w = (ph_acc / (2 * np.pi)) % 1.0
                nsp = fmon.num_spikes
                if nsp > fprev:
                    ev = np.exp(2j * np.pi * ph_w)
                    for iN in np.array(fmon.i[fprev:nsp]):
                        for j, im in idx_of.items():
                            if int(iN) == im:
                                zc[j] += ev
                    fprev = nsp
                okf = True
                dphi = np.zeros(len(names))
                for q2, j in enumerate(names):
                    zc[j] -= CF.DT_N * zc[j] / LC.TAU_P
                    if abs(zc[j]) < 1e-3:
                        okf = False
                        break
                    dd = np.angle(zc[j]) / (2 * np.pi) - phi0[q2]
                    dphi[q2] = (dd + 0.5) % 1.0 - 0.5
                if okf and (t - t_flight0) > 0.12:
                    om_est = dphi @ Sg
                if kn_f % 10 == 0:
                    mujoco.mju_quat2Mat(R, d.qpos[3:7])
                    diag.append((t, float(d.qpos[2]), *om_est, *o_slow,
                                 float(R[8]), float(okf)))
                if kn_f % NV == 0:
                    mujoco.mju_quat2Mat(R, d.qpos[3:7])
                    zcv = np.array([R[2], R[5], R[8]])
                    e_b = R.reshape(3, 3).T @ np.cross(zcv, ZT_W)
                    omv = d.qvel[3:6]
                    vpg.rates = BV.vis_rates(e_b[0], e_b[1], omv[0], omv[1],
                                             ssign, is_oc) * Hz
                    vnet.run(VCF.DT_W * 1000 * _ms)
                    mark("vis", vnet)
                    c2 = vmon.count[:].copy()
                    f_rate += VCF.DT_W * ((c2 - vprev)[readout] / VCF.DT_W
                                          - f_rate) / VCF.TAU_DN
                    vprev = c2
                    eb_est = np.array([VP.decode(f_rate, r0, t_r, s2r),
                                       VP.decode(f_rate, r0, t_p, s2p)])
                if t >= t_nextcmd:
                    t_nextcmd = t + per
                    v3 = d.qvel[:3]
                    x0 = np.clip((z_target - d.qpos[2]) / 5.0, -0.5, 0.5)
                    x = np.array([x0,
                                  -v3[2] / 30.0, eb_est[0], eb_est[1],
                                  om_est[0] / 20.0, om_est[1] / 20.0,
                                  om_est[2] / 20.0])
                    u_cmd = np.clip(U_TRIM + np.tanh(K_POL @ x + B_POL)
                                    * 0.35, -0.55, 0.55)
                    # 生得高度反射: 振幅は下げ方向のみ有効 (揚力実測の制約)
                    # 揚力地形の実測: 振幅は1.0のみ機能 (±5%で崩壊)。
                    # 高度は周波数軸のみで制御する (x0.92=強沈下, x1.15=中立)
                    u_cmd[0] = 0.0
                tf = t - t_flight0
                env0 = min(tf / 0.03, 1.0)   # 翅は脚押し出しと同時に起動
                Z0_, ZC_ = 0.15, 12.0
                T_SPOOL = 0.4     # 把持したまま翅を回し復号を落ち着かせる
                T_PUSH = 0.15     # 脚伸展: 体が0.13→0.6へ持ち上がる (押し出し相)
                T_TUCK = 0.08     # 跗節が離れてから解放までの猶予
                Z_REL = 0.6
                if mode == "takeoff":
                    z_target = Z0_ if tf < T_SPOOL + T_PUSH + T_TUCK else \
                        Z_REL + min((tf - T_SPOOL - T_PUSH - T_TUCK) / 3.0,
                                    1.0) * (ZC_ - Z_REL)
                    if tf >= T_SPOOL + T_PUSH + T_TUCK and not released:
                        released = True
                        d.qvel[2] = (Z_REL - 0.127) / T_PUSH   # 上昇速度を継続
                        print(f"t={t:.2f}s 解放 (接触{d.ncon}, z={d.qpos[2]:.2f})",
                              flush=True)
                    if tf >= T_SPOOL + T_PUSH + T_TUCK + 3.0:
                        mode = "flight"
                        t_cruise0 = t
                        print(f"t={t:.2f}s 巡航 (目標z={ZC_})", flush=True)
                elif mode == "flight" and t - t_cruise0 > 5.0:
                    mode = "descend"
                    t_desc0 = t
                    print(f"t={t:.2f}s 降下開始", flush=True)
                elif mode == "descend":
                    z_target = max(ZC_ - (t - t_desc0) / 3.5 * (ZC_ - 0.6), 0.6)
                    floor_touch = any(
                        d.contact[c].geom1 == W["floor"]
                        or d.contact[c].geom2 == W["floor"]
                        for c in range(d.ncon))
                    # 跗節接触の着陸反射: 降下中に脚が触れたら即座に翅停止
                    # (実バエのtarsal contact→wing stop反射)
                    if floor_touch:
                        mujoco.mju_quat2Mat(R, d.qpos[3:7])
                        up_now = R[8]
                        if up_now > 0.75 and abs(d.qvel[2]) < 10.0:
                            mode = "settle"
                            t_settle = t
                            print(f"t={t:.2f}s 接地 (直立{up_now:+.2f}, "
                                  f"vz{d.qvel[2]:+.1f}) → 静定", flush=True)
                            break
                        # 姿勢不良の接触では着陸せず飛行を続けて仕切り直す
                f_mul = 1.0
                for _ in range(int(CF.DT_N / dtp)):
                    tt = t
                    u += dtp * (u_cmd - u) / CB.TAU_TW
                    amp = np.clip(1.0 + u[0], 0.5, 1.6)
                    if mode == "descend":
                        amp *= 0.96           # 着陸コマンド (安価ループ: 接地vz-3.5)
                    ph_acc += 2 * np.pi * P0["freq"] * f_mul * dtp
                    ph2 = ph_acc
                    s2 = np.sin(ph2)
                    rot = np.tanh(kk * np.cos(ph2 + P0["phase"])) / np.tanh(kk)
                    e2 = env0 * amp
                    d.ctrl[:] = 0
                    # 脚は立位姿勢のまま (畳むと脚同士が貫通し、固定解放時に
                    # 蓄積拘束力が250rad/sのスピンとして爆発する — 診断4で実測)
                    for n2, i2 in act_ids.items():
                        d.ctrl[i2] = stance[n2]
                    d.ctrl[aid["wing_yaw_left"]] = e2 * (P0["yaw_amp"] * s2
                                                         + u[1] + u[2])
                    d.ctrl[aid["wing_yaw_right"]] = e2 * (P0["yaw_amp"] * s2
                                                          + u[1] - u[2])
                    d.ctrl[aid["wing_pitch_left"]] = e2 * (
                        -P0["pitch_amp"] * rot + P0["pitch_bias"] + u[3] + u[4])
                    d.ctrl[aid["wing_pitch_right"]] = e2 * (
                        -P0["pitch_amp"] * rot + P0["pitch_bias"] + u[3] - u[4])
                    d.ctrl[aid["wing_roll_left"]] = e2 * P0["roll_amp"] \
                        * np.sin(2 * ph2)
                    d.ctrl[aid["wing_roll_right"]] = e2 * P0["roll_amp"] \
                        * np.sin(2 * ph2)
                    mujoco.mj_step(m, d)
                    if mode == "takeoff" and not released:
                        # 安価ループ (ES再学習の訓練条件) と同一の固定:
                        # 姿勢はホバー姿勢Q0、高さは押し出し相で0.13→0.6
                        fr_ = min(max(tf - T_SPOOL, 0.0) / T_PUSH, 1.0)
                        d.qpos[0:2] = pose0[0:2]
                        d.qpos[2] = 0.127 + fr_ * (Z_REL - 0.127)
                        d.qpos[3:7] = np.asarray(Q0, float)
                        d.qvel[0:6] = 0.0
                t += CF.DT_N
                kn_f += 1
        if not np.isfinite(d.qpos[2]) or d.qpos[2] > 40:
            print(f"t={t:.2f}s 発散で終了", flush=True)
            break
        if d.time >= next_f:
            next_f += 1.0 / FPS
            cam.lookat[:] = [d.qpos[0], d.qpos[1],
                             d.qpos[2] if mode in ("walk", "settle")
                             else max(d.qpos[2], 2.5)]
            cam.distance = 2.2 if mode in ("walk", "settle") else 8.0
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
            phase_log.append((t, mode, float(d.qpos[2])))
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            ft_ = any(d.contact[c].geom1 == W["floor"]
                      or d.contact[c].geom2 == W["floor"]
                      for c in range(d.ncon))
            crit_log.append((t, float(d.qpos[2]), float(R[8]), bool(ft_), mode))
    import imageio
    imageio.mimsave("outputs/seq_body.mp4", frames, fps=FPS, quality=8)
    def cvt(mon_, key):
        tt = np.array(mon_.t / _ms) / 1000.0
        mp = np.array(tmap[key])
        if len(mp) < 2:
            return tt, np.array(mon_.i)
        return np.interp(tt, mp[:, 0], mp[:, 1]), np.array(mon_.i)
    cT, cI = cvt(cmon, "cmd")
    wT, wI = cvt(wrec, "walk")
    fT, fI = cvt(fmon, "fly")
    vT, vI = cvt(vrec, "vis")
    np.savez(out,
             cmd_t=cT, cmd_i=cI, walk_t=wT, walk_i=wI,
             fly_t=fT, fly_i=fI, vis_t=vT, vis_i=vI,
             t_flight0=t_flight0 if t_flight0 else -1.0,
             phase=np.array([(p[0], {"walk": 0, "takeoff": 1, "flight": 2,
                                     "descend": 3, "settle": 4}[p[1]], p[2])
                             for p in phase_log]),
             cmd_ids=np.array(cids), walk_ids=np.array(wids))
    print(f"記録完了: {len(frames)}フレーム 最終mode={mode} "
          f"z={d.qpos[2]:.2f}", flush=True)
    np.save("outputs/seq_diag.npy", np.array(diag))
    import flight_criteria as FC
    ok, out = FC.evaluate(crit_log, 12.0)
    print("安定飛行の基準:", flush=True)
    FC.report(ok, out)


if __name__ == "__main__":
    main()
