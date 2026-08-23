#!/usr/bin/env python
"""統合39改訂3: 歩行 → [カット] → 安定飛行 (実証済みエンジン) → 軟着陸 → 静定 → 歩行。

飛行は connectome_bioflight.fly_engine (統合31で10秒安定飛行を実証した fly() と
同一の制御・復号ループ)。歩行は統合38c の反射歩容。遷移は指令DN群の発火。
地上離陸は未解決 (README追記35) のため、飛行区間は空中 (z=12, ホバー姿勢) から開始し、
映像上は区間カットとして明示する。
"""
import os, json
import numpy as np
import mujoco
from brian2 import Hz, ms as _ms, prefs, SpikeMonitor
prefs.codegen.target = "numpy"
import walk_base as WB
import walk_reflex as WR
import walk_drive as WD
import vnc_model
import locked_circuit as LC
import connectome_bioflight as CB
import vision_pop as VP
import brain_visual as BV
import vision_circuit_flight as VCF
import flight_criteria as FC
from behavior_sequence import build_command_group, CMD_IDS
from closed_loop import sensory_pools
from closed_loop4 import compensation_factors

FPS = 60
DT_WALK = 0.002
ZC = 12.0


def main():
    T_WALK1, T_CRUISE, T_DESC, T_SETTLE, T_WALK2 = 2.2, 6.0, 3.5, 0.6, 3.0
    cnet, cmon, cpg, csl, cids = build_command_group()
    # ---- 歩行 (統合38c) ----
    W = WB.env(); m = W["m"]
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
            if k in sc: return k
        return "other"
    KIND = {b: sn_kind(b) for b in sn_all}
    import integrate_measured as IMm
    mp = IMm.build_measured_pools()
    excit = {int(r.bodyid): float(IMm.E_MAX ** (1.0 - r.pct)) for _, r in mp.iterrows()}
    tonic = {int(r.bodyid): float(IMm.ITN_MAX * max(0.0, (IMm.TONIC_CUTOFF - r.pct) / IMm.TONIC_CUTOFF)) for _, r in mp.iterrows()}
    wnet, wmon, wids, widx, wpg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=150.0, sensory_bodyids=sn_all,
        excitability=excit, tonic_mv=tonic)
    wrec = SpikeMonitor(wmon.source, name="walk_rec"); wnet.add(wrec)
    LEGS = WR.LEGS; LEG2SUF = WR.LEG2SUF
    leg_slices, kind_masks = {}, {}; at = 0
    for leg in LEGS:
        legk = [b for b in sn_by_leg[leg] if b in widx]
        leg_slices[leg] = slice(at, at + len(legk))
        for kd in ("club", "hook", "other"):
            kind_masks[(leg, kd)] = np.array([i for i, b in enumerate(legk) if KIND[b] == kd], dtype=int)
        cl = [i for i, b in enumerate(legk) if KIND[b] == "claw"]
        kind_masks[(leg, "claw_flex")] = np.array([i for i in cl if legk[i] % 2 == 0], dtype=int)
        kind_masks[(leg, "claw_ext")] = np.array([i for i in cl if legk[i] % 2 == 1], dtype=int)
        at += len(legk)
    sens_comp, motor_comp = compensation_factors(sn_by_leg)
    pools = WD.flybody_pools(widx, W)
    F = {k: np.zeros(len(v)) for k, v in pools.items()}
    legjq = {leg: [m.jnt_qposadr[m.actuator_trnid[act_ids[n]][0]] for n in an if n.endswith(f"{s}_{sd}")] for leg, (s, sd) in LEG2SUF.items()}
    legdq = {leg: [m.jnt_dofadr[m.actuator_trnid[act_ids[n]][0]] for n in an if n.endswith(f"{s}_{sd}")] for leg, (s, sd) in LEG2SUF.items()}
    leggeo = {leg: [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(f"_{s}_{sd}_collision")] for leg, (s, sd) in LEG2SUF.items()}
    q_neutral = {leg: np.array([stance[n] for n in an if n.endswith(f"{LEG2SUF[leg][0]}_{LEG2SUF[leg][1]}")]) for leg in LEGS}
    # ---- 飛行 (実証済みエンジン) ----
    dec = CB.calibrate()
    VP.TAU_DN = VCF.TAU_DN
    vnet, vmon, vpg, ssign, is_oc, readout = VP.build()
    vrec = SpikeMonitor(vmon.source, name="vis_rec"); vnet.add(vrec)
    # ---- 状態・記録 ----
    d = mujoco.MjData(m); WB.reset_standing(m, d); WB.settle(m, d, W, 0.3)
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.elevation, cam.azimuth = -10, 105
    R = np.zeros(9)
    frames, phase_log, crit_log = [], [], []
    tmap = {k: [] for k in ("cmd", "walk", "vis")}
    state = {"t": 0.0, "mode": "walk", "next_f": 0.0}
    def floor_touch():
        return any(d.contact[c].geom1 == W["floor"] or d.contact[c].geom2 == W["floor"] for c in range(d.ncon))
    def record_frame(t, mode):
        if t >= state["next_f"]:
            state["next_f"] += 1.0 / FPS
            cam.lookat[:] = [d.qpos[0], d.qpos[1], d.qpos[2] if mode in ("walk", "settle") else max(d.qpos[2], 2.5)]
            cam.distance = 2.2 if mode in ("walk", "settle") else 8.0
            renderer.update_scene(d, cam); frames.append(renderer.render())
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            phase_log.append((t, mode, float(d.qpos[2])))
            crit_log.append((t, float(d.qpos[2]), float(R[8]), bool(floor_touch()), mode))
    def run_cmd(t, rates):
        cpg.rates = rates * Hz; cnet.run(DT_WALK * 1000 * _ms)
        tmap["cmd"].append((float(cnet.t / _ms) / 1000.0, t))
    prev_w = np.zeros(wmon.source.N); rates_vec = np.zeros(at)
    fref = {"v": 1e-6}; prev_cmd = {}
    def walk_step(t):
        fr6 = np.zeros(6); fbuf = np.zeros(6)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if g1 != W["floor"] and g2 != W["floor"]: continue
            go = g2 if g1 == W["floor"] else g1
            mujoco.mj_contactForce(m, d, c, fbuf)
            for li, leg in enumerate(LEGS):
                if go in leggeo[leg]: fr6[li] += abs(fbuf[0])
        loads = dict(zip(LEGS, fr6)); fref["v"] = max(fref["v"], max(loads.values()))
        for leg in LEGS:
            qs = np.array([d.qpos[qa] for qa in legjq[leg]]); dqs = np.array([d.qvel[qa] for qa in legdq[leg]])
            dev_s = float((qs - q_neutral[leg]).sum()); vel = np.abs(dqs).sum(); flexing = float(np.mean(dqs))
            sl2 = leg_slices[leg]; base = np.full(sl2.stop - sl2.start, WR.R_TONIC); mk = kind_masks
            base[mk[(leg, "claw_flex")]] += WR.KP * max(dev_s, 0.0)
            base[mk[(leg, "claw_ext")]] += WR.KP * max(-dev_s, 0.0)
            base[mk[(leg, "club")]] += 8.0 * WR.KV * vel
            base[mk[(leg, "hook")]] += 60.0 * max(flexing, 0.0)
            base[mk[(leg, "other")]] += WR.KL * loads[leg] / fref["v"]
            rates_vec[sl2] = np.clip(base * sens_comp[leg], 0, WR.R_MAX)
        wpg.rates = rates_vec * Hz
        wnet.run(DT_WALK * 1000 * _ms); tmap["walk"].append((float(wnet.t / _ms) / 1000.0, t))
        cnt = wmon.count[:].copy(); dspk = cnt - prev_w; prev_w[:] = cnt
        qcmd = dict(stance)
        for (act2, dr), lst in pools.items():
            f = F[(act2, dr)]
            for q2, (ni, g, tau, fmax) in enumerate(lst):
                f[q2] += dspk[ni] * g; f[q2] = min(f[q2], fmax); f[q2] -= DT_WALK * f[q2] / tau
            seg, sd = act2.split("_")[1], act2.split("_")[2]
            leg = [k for k, v in LEG2SUF.items() if v == (seg, sd)][0]
            scale = WR.EXT_SCALE if dr < 0 else 1.0
            qcmd[act2] = qcmd.get(act2, stance[act2]) + dr * scale * WR.K_WALK * WD.GAIN_JOINT[act2.split("_")[0]] * motor_comp[leg] * float(f.sum())
        for n2, i2 in act_ids.items():
            lo, hi = m.actuator_ctrlrange[i2]
            v2 = np.clip(qcmd[n2], stance[n2] - WR.Q_CLAMP, stance[n2] + WR.Q_CLAMP)
            pv = prev_cmd.get(n2, stance[n2]); v2 = np.clip(v2, pv - WR.SLEW, pv + WR.SLEW)
            prev_cmd[n2] = v2; d.ctrl[i2] = np.clip(v2, lo, hi)
        for w2 in ("wing_yaw_left", "wing_yaw_right", "wing_pitch_left", "wing_pitch_right", "wing_roll_left", "wing_roll_right"):
            d.ctrl[W["aid"][w2]] = 0
        for _ in range(int(DT_WALK / m.opt.timestep)): mujoco.mj_step(m, d)
    # ===== 区間A: 歩行 (MDN) =====
    t = 0.0
    cmd_rates = np.zeros(len(cids))
    print("区間A: 歩行", flush=True)
    gf_seen = 0; took = False
    while t < T_WALK1 + 2.0 and not took:
        cmd_rates[:] = 0; cmd_rates[csl["MDN"]] = 250.0
        if t >= T_WALK1: cmd_rates[csl["GF"]] = 800.0
        run_cmd(t, cmd_rates)
        gf_now = int(cmon.count[:][csl["GF"]].sum())
        walk_step(t); t += DT_WALK; record_frame(t, "walk")
        if gf_now > gf_seen + 1:
            took = True; print(f"t={t:.2f}s GF発火{gf_now} → [カット] 空中開始 z={ZC}", flush=True)
    # ===== 区間B: 飛行 (空中開始、実証済みエンジン) =====
    t_cut = t
    # カット: 関節状態をデフォルトへ戻す (歩行後の脚・翅関節状態のまま飛行エンジンへ
    # 渡すと218rad/sの衝撃が出る — cut_test.pyで実測) 。体位置・姿勢は空中・ホバー姿勢
    d_def = mujoco.MjData(m); mujoco.mj_resetData(m, d_def)
    d.qpos[7:] = d_def.qpos[7:]
    if d.act.size: d.act[:] = 0
    d.ctrl[:] = 0
    d.qpos[2] = ZC; d.qpos[3:7] = np.asarray(W["Q0"], float); d.qvel[:] = 0.0
    d.qvel[3:6] = np.random.default_rng(1).normal(0, 1.0, 3); mujoco.mj_forward(m, d)
    vprev = {"c": vmon.count[:].copy()}; fl_mode = {"m": "flight"}; diag = []
    def z_fn(tn):
        return ZC if tn < T_CRUISE else max(ZC - (tn - T_CRUISE) / T_DESC * (ZC - 0.6), 0.6)
    def land_amp(tn):
        return 1.0 if tn < T_CRUISE else 0.96
    def frame_cb(d_, t_abs):
        tn = t_abs - t_cut
        md = "flight" if tn < T_CRUISE else "descend"
        fl_mode["m"] = md
        if int(t_abs / 0.016) != int((t_abs - CB.CF.DT_N) / 0.016):   # ~60Hz: 視覚net表示用
            mujoco.mju_quat2Mat(R, d_.qpos[3:7]); zcv = np.array([R[2], R[5], R[8]])
            e_b = R.reshape(3, 3).T @ np.cross(zcv, W["ZT_W"]); omv = d_.qvel[3:6]
            vpg.rates = BV.vis_rates(e_b[0], e_b[1], omv[0], omv[1], ssign, is_oc) * Hz
            vnet.run(16 * _ms); tmap["vis"].append((float(vnet.t / _ms) / 1000.0, t_abs))
            cmd_rates[:] = 0; run_cmd(t_abs, cmd_rates)
        record_frame(t_abs, md)
    def land_cb(d_, tn):
        if tn < T_CRUISE: return False
        if floor_touch():
            mujoco.mju_quat2Mat(R, d_.qpos[3:7])
            if R[8] > 0.75 and abs(d_.qvel[2]) < 10.0:
                print(f"t={t_cut + tn:.2f}s 接地 (直立{R[8]:+.2f}, vz{d_.qvel[2]:+.1f}) → 静定", flush=True)
                return True
        return False
    print("区間B: 飛行 (エンジン=fly_engine)", flush=True)
    t_fly, reason, (fl_t, fl_i) = CB.fly_engine(
        d, dec, T=T_CRUISE + T_DESC + 3.0, z_fn=z_fn, land_amp=land_amp, land_cb=land_cb,
        frame_cb=frame_cb, diag=diag, t0=t_cut, leg_stance=(W["leg_act"], W["stance_q"]))
    t = t_cut + t_fly
    print(f"飛行終了: {reason} ({t_fly:.2f}s)", flush=True)
    # ===== 区間C: 静定 → 歩行 =====
    t_settle = t; ok_settle = False
    while t - t_settle < 2.5:
        d.ctrl[:] = 0; d.ctrl[W["leg_act"]] = W["stance_q"]
        for _ in range(int(DT_WALK / m.opt.timestep)): mujoco.mj_step(m, d)
        t += DT_WALK; cmd_rates[:] = 0; run_cmd(t, cmd_rates); record_frame(t, "settle")
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        if t - t_settle > T_SETTLE and R[8] > 0.85:
            ok_settle = True; print(f"t={t:.2f}s 静定完了 (直立{R[8]:+.2f}) → 歩行再開", flush=True); break
    if not ok_settle: print(f"t={t:.2f}s 静定失敗 (直立{R[8]:+.2f})", flush=True)
    t_w2 = t
    while t - t_w2 < T_WALK2:
        cmd_rates[:] = 0; cmd_rates[csl["MDN"]] = 250.0; run_cmd(t, cmd_rates)
        walk_step(t); t += DT_WALK; record_frame(t, "walk")
    # ===== 保存・判定 =====
    import imageio
    imageio.mimsave("outputs/seq_body.mp4", frames, fps=FPS, quality=8)
    def cvt(mon_, key):
        tt = np.array(mon_.t / _ms) / 1000.0; mp_ = np.array(tmap[key])
        return (np.interp(tt, mp_[:, 0], mp_[:, 1]) if len(mp_) > 1 else tt), np.array(mon_.i)
    cT, cI = cvt(cmon, "cmd"); wT, wI = cvt(wrec, "walk"); vT, vI = cvt(vrec, "vis")
    np.savez("outputs/seq_record.npz", cmd_t=cT, cmd_i=cI, walk_t=wT, walk_i=wI,
             fly_t=fl_t, fly_i=fl_i, vis_t=vT, vis_i=vI, t_flight0=t_cut,
             phase=np.array([(p[0], {"walk": 0, "takeoff": 1, "flight": 2, "descend": 3, "settle": 4}[p[1]], p[2]) for p in phase_log]),
             cmd_ids=np.array(cids), walk_ids=np.array(wids))
    np.save("outputs/seq_diag.npy", np.array(diag))
    print(f"記録完了: {len(frames)}フレーム 最終z={d.qpos[2]:.2f}", flush=True)
    ok, out = FC.evaluate(crit_log, ZC, startup_skip=(t_cut, 1.0))
    print("安定飛行の基準:", flush=True); FC.report(ok, out)


if __name__ == "__main__":
    main()
