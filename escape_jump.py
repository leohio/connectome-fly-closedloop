#!/usr/bin/env python
"""逃避ジャンプ: 巨大線維(GF=DNp01)パルス → TTM → 中脚伸展 → 跳躍を試す。

立位(感覚フィードバックのみ、MDNなし)のハエに、t=0.5s で GF を 60ms 刺激する。
TTMn(跳躍筋MN)のスパイクを中脚(LM/RM)の Femur/Tibia 伸展に強くマップし、
体高 z の跳ね上がりを観測する。
"""
import numpy as np
import pandas as pd
from brian2 import ms, Hz, prefs

import vnc_model
from integrate import build_pools, DT_WIN, PHYS_DT, TAU_ACT, R0
from closed_loop import sensory_pools
from closed_loop4 import compensation_factors, GAIN, EXT_SCALE, Q_CLAMP, ACT_MAX, \
    R_TONIC, KP, KV, KL, LEGS

prefs.codegen.target = "numpy"

T_TOTAL = 1.2
T_PULSE = (0.5, 0.56)
GF_RATE = 400
SLEW = 0.12          # ジャンプは速い動作なのでスルーレート緩和
TTM_GAIN = 1.2       # TTM→中脚伸展の専用ゲイン


def main():
    from flygym import Fly, Camera, SingleFlySimulation

    sn_by_leg = sensory_pools()
    sens_comp, motor_comp = compensation_factors(sn_by_leg)
    neurons = pd.read_csv("../vnc-connectome/downloads/traced-neurons.csv")
    gf_ids = neurons[neurons.type == "DNp01"].bodyId.tolist()
    mns3 = pd.read_csv("../vnc-connectome/downloads/elife-96084-supp3-v1.csv",
                       encoding="latin1")
    ttm_ids = mns3[mns3.type.astype(str).str.startswith("TTMn")].bodyid.tolist()
    print("GF:", gf_ids, " TTMn:", ttm_ids)

    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    ctrl_all = sn_all + gf_ids          # 感覚 + GF を可変レート群に
    net, mon, ids, idx, pg = vnc_model.make_network(
        [], r_stim_hz=0, sensory_bodyids=ctrl_all)
    kept = [b for b in ctrl_all if b in idx]
    pos_of = {b: i for i, b in enumerate(kept)}
    leg_slices = {leg: np.array([pos_of[b] for b in bs if b in pos_of])
                  for leg, bs in sn_by_leg.items()}
    gf_slice = np.array([pos_of[b] for b in gf_ids if b in pos_of])

    pools = build_pools(idx)
    ttm_members = np.array([idx[b] for b in ttm_ids if b in idx])

    fly = Fly(init_pose="tripod", control="position", enable_adhesion=False)
    cam = Camera(fly=fly, play_speed=0.1)
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=PHYS_DT)
    obs, _ = sim.reset()
    q_neutral = obs["joints"][0].copy()
    joint_names = list(fly.actuated_joints)
    jidx = {}
    for (jname, d) in pools:
        m = [k for k, n in enumerate(joint_names) if jname in n]
        if m:
            jidx[(jname, d)] = m[0]
    ttm_joints = [k for k, n in enumerate(joint_names)
                  if ("LMFemur" in n or "RMFemur" in n or
                      "LMTibia" in n or "RMTibia" in n) and "roll" not in n]
    sense_j = {leg: [k for k, n in enumerate(joint_names)
                     if (f"{leg}Femur" in n or f"{leg}Tibia" in n)
                     and "roll" not in n] for leg in LEGS}
    contact_names = [str(s) for s in fly.contact_sensor_placements]
    leg_contact = {leg: [k for k, n in enumerate(contact_names) if leg in n]
                   for leg in LEGS}

    n_win = int(T_TOTAL / DT_WIN)
    act = {k: 0.0 for k in pools}
    ttm_act = 0.0
    q_cmd_prev = q_neutral.copy()
    prev_count = np.zeros(len(ids))
    rates_vec = np.zeros(len(kept))
    f_ref = 1e-6
    log_z, log_ttm = [], []

    for wi in range(n_win):
        tnow = wi * DT_WIN
        qs, dqs = obs["joints"][0], obs["joints"][1]
        cf = obs["contact_forces"]
        for leg in LEGS:
            F = float(np.linalg.norm(cf[leg_contact[leg]], axis=1).sum()) if len(leg_contact[leg]) else 0.0
            f_ref = max(f_ref, F)
            js = sense_j[leg]
            dev = np.abs(qs[js] - q_neutral[js]).sum()
            vel = np.abs(dqs[js]).sum()
            r = (R_TONIC + KP * dev + KV * vel + KL * (F / f_ref)) * sens_comp[leg]
            rates_vec[leg_slices[leg]] = np.clip(r, 0, 250)
        rates_vec[gf_slice] = GF_RATE if T_PULSE[0] <= tnow < T_PULSE[1] else 0.0
        pg.rates = rates_vec * Hz
        net.run(2 * ms)
        count = mon.count[:]
        d_spk = count - prev_count
        prev_count = count.copy()
        q = q_neutral.copy()
        for key, members in pools.items():
            rate = d_spk[members].sum() / len(members) / DT_WIN
            act[key] += DT_WIN * ((rate / R0) - act[key]) / TAU_ACT
            act[key] = min(max(act[key], 0.0), ACT_MAX)
        ttm_rate = d_spk[ttm_members].sum() / max(len(ttm_members), 1) / DT_WIN
        ttm_act += DT_WIN * ((ttm_rate / R0) - ttm_act) / (TAU_ACT / 2)
        ttm_act = min(max(ttm_act, 0.0), 3.0)
        for (jname, d), k in jidx.items():
            base = jname.split("joint_")[1][2:]
            leg = jname.replace("joint_", "")[:2]
            scale = EXT_SCALE if d < 0 else 1.0
            q[k] += GAIN[base] * d * scale * motor_comp[leg] * act[(jname, d)]
        for k in ttm_joints:                      # TTM: 中脚を強伸展
            q[k] -= TTM_GAIN * ttm_act
        q = np.clip(q, q_neutral - Q_CLAMP, q_neutral + Q_CLAMP)
        q = np.clip(q, q_cmd_prev - SLEW, q_cmd_prev + SLEW)
        q_cmd_prev = q.copy()
        for _ in range(int(DT_WIN / PHYS_DT)):
            obs, *_ = sim.step({"joints": q})
        sim.render()
        log_z.append(obs["fly"][0][2])
        log_ttm.append(ttm_act)

    cam.save_video("outputs/escape_jump.mp4")
    z = np.array(log_z)
    np.savez("outputs/escape_jump_log.npz", z=z, ttm=np.array(log_ttm))
    pre = z[:int(T_PULSE[0] / DT_WIN)].mean()
    post_max = z[int(T_PULSE[0] / DT_WIN):].max()
    print(f"\nbody z: pre-pulse mean {pre:.2f} -> post-pulse max {post_max:.2f} "
          f"(jump +{post_max - pre:.2f})")
    print(f"TTM act max: {max(log_ttm):.2f}")
    print("saved outputs/escape_jump.mp4")


if __name__ == "__main__":
    main()
