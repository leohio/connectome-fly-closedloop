#!/usr/bin/env python
"""完全統合プロトタイプ: MANC腹髄スパイキング回路が NeuroMechFly の身体を直接駆動する。

  MDNポアソン刺激 (脳からの後退歩行指令のモデル)
    → MANC 23,188ニューロン LIF (実測配線, Brian2)
    → 脚運動ニューロンのスパイク
    → 筋活動 (拮抗筋プールごとに低域通過, τ=40ms)
    → 関節角 (拮抗差 × ゲイン)
    → MuJoCo 物理 (NeuroMechFly)

  2ms 窓で神経と物理を交互に実行する。CPG・事前プログラム歩行は一切使わない。
"""
import numpy as np
import pandas as pd
from brian2 import ms, prefs

import vnc_model

prefs.codegen.target = "numpy"

DT_WIN = 2e-3        # 神経⇔物理の交換窓 [s]
T_TOTAL = 0.6        # 総時間 [s]
PHYS_DT = 1e-4
MDN_RATE = 300       # 強めに駆動
TAU_ACT = 0.040      # 筋活動時定数 [s]
R0 = 15.0            # 発火率正規化 [Hz]

# 筋ターゲット → (関節, 方向) マッピング
MUSCLE2JOINT = {
    "Ti flexor":                ("Tibia", +1), "Acc. ti flexor": ("Tibia", +1),
    "Ti extensor":              ("Tibia", -1),
    "Tr flexor":                ("Femur", +1), "Acc. tr flexor": ("Femur", +1),
    "Tr extensor":              ("Femur", -1), "Tergotr.": ("Femur", -1),
    "Sternotrochanter":         ("Femur", -1),
    "Sternal anterior rotator": ("Coxa", +1),
    "Sternal posterior rotator": ("Coxa", -1),
    "Pleural remotor/abductor": ("Coxa", -1),
    "Sternal adductor":         ("Coxa", -1),
    "ltm": ("Tarsus1", +1), "ltm1-tibia": ("Tarsus1", +1), "ltm2-femur": ("Tarsus1", +1),
}
GAIN = {"Coxa": 0.5, "Femur": 0.6, "Tibia": 0.8, "Tarsus1": 0.4}
SEG2LEG = {"T1": "F", "T2": "M", "T3": "H"}
SIDE2LEG = {"LHS": "L", "RHS": "R"}


def build_pools(idx):
    """(joint_name, direction) -> ニューロンindex配列"""
    mns = vnc_model.leg_mn_table()
    pools = {}
    for _, r in mns.iterrows():
        if r.target not in MUSCLE2JOINT or r.bodyid not in idx:
            continue
        joint, d = MUSCLE2JOINT[r.target]
        leg = SIDE2LEG[r.soma_side] + SEG2LEG[r.soma_neuromere]
        key = (f"joint_{leg}{joint}", d)
        pools.setdefault(key, []).append(idx[r.bodyid])
    return {k: np.array(v) for k, v in pools.items()}


def main():
    from flygym import Fly, Camera, SingleFlySimulation

    print("building VNC network...")
    net, mon, ids, idx = vnc_model.make_network(
        vnc_model.MDN_BODYIDS, r_stim_hz=MDN_RATE)
    pools = build_pools(idx)
    print(f"pools: {len(pools)} (脚×関節×方向)")

    fly = Fly(init_pose="stretch", control="position", enable_adhesion=False)
    cam = Camera(fly=fly, play_speed=0.1)
    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=PHYS_DT)
    obs, _ = sim.reset()
    q_neutral = obs["joints"][0].copy()
    joint_names = list(fly.actuated_joints)
    jidx = {}
    for (jname, d), members in pools.items():
        matches = [k for k, n in enumerate(joint_names) if jname in n]
        if matches:
            jidx[(jname, d)] = matches[0]
    print(f"mapped joints: {len(set(j for j,_ in jidx))}/42")

    n_win = int(T_TOTAL / DT_WIN)
    act = {k: 0.0 for k in pools}
    prev_count = np.zeros(len(ids))
    log_act, log_q, log_pos = [], [], []

    for wi in range(n_win):
        net.run(2 * ms)                             # 神経 2ms
        count = mon.count[:]
        d_spk = count - prev_count
        prev_count = count.copy()
        q = q_neutral.copy()
        for key, members in pools.items():
            rate = d_spk[members].sum() / len(members) / DT_WIN  # pool平均Hz
            a = act[key]
            act[key] = a + DT_WIN * ((rate / R0) - a) / TAU_ACT
        for (jname, d), k in jidx.items():
            base = jname.split("joint_")[1][2:]     # Coxa/Femur/...
            q[k] += GAIN[base] * d * act[(jname, d)]
        for _ in range(int(DT_WIN / PHYS_DT)):      # 物理 2ms
            obs, *_ = sim.step({"joints": q})
        sim.render()
        log_act.append({f"{j}|{d}": act[(j, d)] for (j, d) in pools})
        log_q.append(q.copy())
        log_pos.append(obs["fly"][0].copy())
        if wi % 50 == 0:
            print(f"  t={wi*DT_WIN:.2f}s spikes={int(count.sum())} "
                  f"pos={np.round(obs['fly'][0][:2], 2)}")

    cam.save_video("outputs/integrated_mdn.mp4")
    np.savez("outputs/integrated_log.npz",
             act=pd.DataFrame(log_act).values,
             act_cols=list(pd.DataFrame(log_act).columns),
             q=np.array(log_q), pos=np.array(log_pos))
    pos = np.array(log_pos)
    print(f"\ntotal spikes: {int(mon.count[:].sum())}")
    print(f"displacement: {np.linalg.norm(pos[-1,:2]-pos[0,:2]):.2f} mm "
          f"(x: {pos[-1,0]-pos[0,0]:+.2f}, y: {pos[-1,1]-pos[0,1]:+.2f})")
    print("saved outputs/integrated_mdn.mp4")


if __name__ == "__main__":
    main()
