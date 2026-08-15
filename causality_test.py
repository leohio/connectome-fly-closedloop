#!/usr/bin/env python
"""因果検証: スパイクが運動を生んでいるかの介入実験。

3条件を同一初期条件・同一時間(1.0s)で比較する(カメラなし高速版):
  A) normal    : 通常の閉ループ(関節指令はMNプールのスパイク数の純関数)
  B) mean-drive: 筋活動を条件Aの時間平均値に固定(スパイクの時間パターンを破壊、
                 平均的な「筋トーヌス」だけ残す)
  C) zero-drive: 神経→筋の結合を切断(act=0、関節は中立姿勢のサーボ保持のみ)

予測: スパイクが運動を生んでいるなら、A で出る足踏み・変位が B/C で消える。
B が A と同等なら「スパイクの平均量」だけが要因で時間構造は無関係、となる。
"""
import numpy as np
from brian2 import ms, Hz, prefs

import vnc_model
from integrate import build_pools, DT_WIN, PHYS_DT, TAU_ACT, R0
from closed_loop import sensory_pools
from closed_loop4 import (compensation_factors, GAIN, EXT_SCALE, Q_CLAMP, SLEW,
                          ACT_MAX, R_TONIC, KP, KV, KL, LEGS, MDN_RATE)

prefs.codegen.target = "numpy"
T_TOTAL = 1.0


def run(mode, mean_act=None):
    from flygym import Fly, SingleFlySimulation

    sn_by_leg = sensory_pools()
    sens_comp, motor_comp = compensation_factors(sn_by_leg)
    sn_all = [b for leg in sn_by_leg.values() for b in leg]
    net, mon, ids, idx, pg = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=MDN_RATE, sensory_bodyids=sn_all)
    kept = [b for b in sn_all if b in idx]
    pos_of = {b: i for i, b in enumerate(kept)}
    leg_slices = {leg: np.array([pos_of[b] for b in bs if b in pos_of])
                  for leg, bs in sn_by_leg.items()}
    pools = build_pools(idx)
    fly = Fly(init_pose="tripod", control="position", enable_adhesion=False)
    sim = SingleFlySimulation(fly=fly, cameras=[], timestep=PHYS_DT)
    obs, _ = sim.reset()
    q_neutral = obs["joints"][0].copy()
    joint_names = list(fly.actuated_joints)
    jidx = {}
    for (jname, d) in pools:
        m = [k for k, n in enumerate(joint_names) if jname in n]
        if m:
            jidx[(jname, d)] = m[0]
    sense_j = {leg: [k for k, n in enumerate(joint_names)
                     if (f"{leg}Femur" in n or f"{leg}Tibia" in n)
                     and "roll" not in n] for leg in LEGS}
    contact_names = [str(s) for s in fly.contact_sensor_placements]
    leg_contact = {leg: [k for k, n in enumerate(contact_names) if leg in n]
                   for leg in LEGS}

    act = {k: 0.0 for k in pools}
    q_cmd_prev = q_neutral.copy()
    prev_count = np.zeros(len(ids))
    rates_vec = np.zeros(len(kept))
    f_ref = 1e-6
    log_pos, log_load, act_hist = [], [], []

    for wi in range(int(T_TOTAL / DT_WIN)):
        qs, dqs = obs["joints"][0], obs["joints"][1]
        cf = obs["contact_forces"]
        loads = {}
        for leg in LEGS:
            F = float(np.linalg.norm(cf[leg_contact[leg]], axis=1).sum()) if len(leg_contact[leg]) else 0.0
            loads[leg] = F
        f_ref = max(f_ref, max(loads.values()))
        for leg in LEGS:
            js = sense_j[leg]
            dev = np.abs(qs[js] - q_neutral[js]).sum()
            vel = np.abs(dqs[js]).sum()
            r = (R_TONIC + KP * dev + KV * vel
                 + KL * (loads[leg] / f_ref)) * sens_comp[leg]
            rates_vec[leg_slices[leg]] = np.clip(r, 0, 250)
        pg.rates = rates_vec * Hz
        net.run(2 * ms)
        count = mon.count[:]
        d_spk = count - prev_count
        prev_count = count.copy()
        for key, members in pools.items():
            rate = d_spk[members].sum() / len(members) / DT_WIN
            act[key] += DT_WIN * ((rate / R0) - act[key]) / TAU_ACT
            act[key] = min(max(act[key], 0.0), ACT_MAX)
        act_hist.append([act[k] for k in sorted(pools)])
        q = q_neutral.copy()
        for (jname, d), k in jidx.items():
            base = jname.split("joint_")[1][2:]
            leg = jname.replace("joint_", "")[:2]
            scale = EXT_SCALE if d < 0 else 1.0
            if mode == "normal":
                a = act[(jname, d)]
            elif mode == "mean":
                a = mean_act[(jname, d)]
            else:                       # zero
                a = 0.0
            q[k] += GAIN[base] * d * scale * motor_comp[leg] * a
        q = np.clip(q, q_neutral - Q_CLAMP, q_neutral + Q_CLAMP)
        q = np.clip(q, q_cmd_prev - SLEW, q_cmd_prev + SLEW)
        q_cmd_prev = q.copy()
        for _ in range(int(DT_WIN / PHYS_DT)):
            obs, *_ = sim.step({"joints": q})
        log_pos.append(obs["fly"][0].copy())
        log_load.append([loads[l] for l in LEGS])
    sim.close()
    pos = np.array(log_pos)
    load = np.array(log_load)
    thr = 0.05 * load.max() if load.max() > 0 else 1
    steps = int(sum((np.diff((load[:, k] > thr).astype(int)) == 1).sum()
                    for k in range(6)))
    disp = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
    mean_acts = {k: float(np.mean([h[i] for h in act_hist]))
                 for i, k in enumerate(sorted(pools))}
    return disp, steps, int(count.sum()), mean_acts


if __name__ == "__main__":
    dispA, stepsA, spkA, meanA = run("normal")
    print(f"A) 通常閉ループ     : 変位 {dispA:.2f}mm  総ステップ {stepsA}  スパイク {spkA}")
    dispB, stepsB, spkB, _ = run("mean", mean_act=meanA)
    print(f"B) 平均值固定(パターン破壊): 変位 {dispB:.2f}mm  総ステップ {stepsB}  スパイク {spkB}")
    dispC, stepsC, spkC, _ = run("zero")
    print(f"C) 神経→筋 切断     : 変位 {dispC:.2f}mm  総ステップ {stepsC}  スパイク {spkC}")
