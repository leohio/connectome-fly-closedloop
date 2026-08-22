#!/usr/bin/env python
"""統合38c: flybodyでの自己受容反射ループ — 律動歩容への第一歩。

統合1-7 (flygym) で足踏みを生んだ反射構成の移植:
  脚ごとの実自己受容SN (MANC entryNerve別) へ、関節偏差・速度・接地荷重から
  計算した発火率を注入 → MANC実配線 → 脚MN → サイズ原理筋 → flybody関節。
  左右・脚間の配線格差はMANCデータから算出した係数で補償 (closed_loop4)。
"""
import sys
import numpy as np
import mujoco
from brian2 import Hz, ms as _ms, prefs
prefs.codegen.target = "numpy"
import vnc_model
import walk_base as WB
import walk_drive as WD
from closed_loop import sensory_pools
from closed_loop4 import compensation_factors

DT_WIN = 0.002
MDN_RATE = 300
R_TONIC, KP, KV, KL, R_MAX = 15.0, 50.0, 0.5, 50.0, 250.0
LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"]
LEG2SUF = {"LF": ("T1", "left"), "LM": ("T2", "left"), "LH": ("T3", "left"),
           "RF": ("T1", "right"), "RM": ("T2", "right"), "RH": ("T3", "right")}


def run(T=3.0, mdn_hz=MDN_RATE, alpha=1.0, video=None):
    W = WB.env()
    m = W["m"]
    an = {n: i for i, n in enumerate(W["leg_names"])}
    act_ids = {n: int(W["leg_act"][i]) for n, i in an.items()}
    stance = {n: float(W["stance_q"][i]) for n, i in an.items()}
    sn_by_leg = sensory_pools()
    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    net, mon, ids, idx, pg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=mdn_hz, sensory_bodyids=sn_all)
    kept = [b for b in sn_all if b in idx]
    leg_slices = {}
    at = 0
    for leg in LEGS:
        nn = len([b for b in sn_by_leg[leg] if b in idx])
        leg_slices[leg] = slice(at, at + nn)
        at += nn
    sens_comp, motor_comp = compensation_factors(sn_by_leg)
    pools = WD.flybody_pools(idx, W)
    print(f"SN {at}本 / プール{len(pools)}方向", flush=True)
    # 脚ごとの関節・接触geom
    legjq = {leg: [m.jnt_qposadr[m.actuator_trnid[act_ids[n]][0]]
                   for n in an if n.endswith(f"{s}_{sd}")]
             for leg, (s, sd) in LEG2SUF.items()}
    leggeo = {}
    for leg, (s, sd) in LEG2SUF.items():
        leggeo[leg] = [g for g in range(m.ngeom)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                           or "").endswith(f"_{s}_{sd}_collision")
                       and any(k in mujoco.mj_id2name(
                           m, mujoco.mjtObj.mjOBJ_GEOM, g)
                           for k in ["tarsus", "tibia", "claw"])]
    F = {k: np.zeros(len(v)) for k, v in pools.items()}
    d = mujoco.MjData(m)
    WB.reset_standing(m, d)
    WB.settle(m, d, W, T=0.3)
    x0, y0 = d.qpos[0], d.qpos[1]
    q_neutral = {leg: np.array([d.qpos[qa] for qa in legjq[leg]])
                 for leg in LEGS}
    prev = np.zeros(mon.source.N)
    rates_vec = np.zeros(at)
    f_ref = 1e-6
    fr6 = np.zeros(6)
    renderer, frames = None, []
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = 1.2, -12, 100
    n_phys = int(DT_WIN / m.opt.timestep)
    contact_log, qlog = [], []
    for wi in range(int(T / DT_WIN)):
        # 接地荷重 (脚ごと)
        fbuf = np.zeros(6)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if g1 != W["floor"] and g2 != W["floor"]:
                continue
            go = g2 if g1 == W["floor"] else g1
            mujoco.mj_contactForce(m, d, c, fbuf)
            fn = abs(fbuf[0])
            for li, leg in enumerate(LEGS):
                if go in leggeo[leg]:
                    fr6[li] += fn
        loads = dict(zip(LEGS, fr6))
        fr6[:] = 0
        f_ref = max(f_ref, max(loads.values()))
        for li, leg in enumerate(LEGS):
            qs = np.array([d.qpos[qa] for qa in legjq[leg]])
            dqs = np.array([d.qvel[m.jnt_dofadr[m.actuator_trnid[
                act_ids[n]][0]]] for n in an
                if n.endswith(f"{LEG2SUF[leg][0]}_{LEG2SUF[leg][1]}")])
            dev = np.abs(qs - q_neutral[leg]).sum()
            vel = np.abs(dqs).sum()
            r = (R_TONIC + alpha * (KP * dev + KV * vel
                 + KL * loads[leg] / f_ref)) * sens_comp[leg]
            rates_vec[leg_slices[leg]] = np.clip(r, 0, R_MAX)
        pg.rates = rates_vec * Hz
        net.run(DT_WIN * 1000 * _ms)
        cnt = mon.count[:].copy()
        dspk = cnt - prev
        prev = cnt
        qcmd = dict(stance)
        for (act, dr), lst in pools.items():
            f = F[(act, dr)]
            for q, (ni, g, tau, fmax) in enumerate(lst):
                f[q] += dspk[ni] * g
                f[q] = min(f[q], fmax)
                f[q] -= DT_WIN * f[q] / tau
            seg, sd = act.split("_")[1], act.split("_")[2]
            leg = [k for k, v in LEG2SUF.items() if v == (seg, sd)][0]
            qcmd[act] = qcmd.get(act, stance[act]) \
                + dr * WD.K_CAL * WD.GAIN_JOINT[act.split("_")[0]] \
                * motor_comp[leg] * float(f.sum())
        for n, i in act_ids.items():
            lo, hi = m.actuator_ctrlrange[i]
            d.ctrl[i] = np.clip(qcmd[n], lo, hi)
        for _ in range(n_phys):
            mujoco.mj_step(m, d)
        if renderer and wi % int(1 / 60 / DT_WIN + 0.5) == 0:
            cam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
        contact_log.append([loads[l] for l in LEGS])
        qlog.append([d.qpos[legjq["LM"][1]], d.qpos[legjq["RM"][1]]])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    q = np.array(qlog)
    C = np.array(contact_log)
    swing = (C < 1e-8).mean(0)
    print(f"MDN={mdn_hz}Hz+反射α={alpha} {T:.1f}s: 直立度={R[8]:+.2f} z={d.qpos[2]:+.3f} "
          f"前進={d.qpos[0]-x0:+.4f} 側方={d.qpos[1]-y0:+.4f}", flush=True)
    print(f"  中脚femur振幅std: L={q[:,0].std():.4f} R={q[:,1].std():.4f} "
          f"L-R相関={np.corrcoef(q[:,0], q[:,1])[0,1]:+.2f}", flush=True)
    print(f"  遊脚率 (接地力ゼロの時間割合) {dict(zip(LEGS, swing.round(2)))}",
          flush=True)
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=60, quality=8)
        print("動画:", video, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "sweep":
        for mdn, al in [(150, 0.25), (150, 0.5), (150, 1.0), (300, 0.25)]:
            run(T=2.0, mdn_hz=mdn, alpha=al)
    else:
        T = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
        mdn = float(sys.argv[2]) if len(sys.argv) > 2 else MDN_RATE
        al = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        vid = sys.argv[4] if len(sys.argv) > 4 else None
        run(T=T, mdn_hz=mdn, alpha=al, video=vid)
