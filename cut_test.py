import sys, numpy as np, mujoco
import walk_base as WB
import locked_circuit as LC
import connectome_bioflight as CB
W = WB.env(); m = W["m"]
dec = CB.calibrate()
def engine_spin(d, label):
    R = np.zeros(9); diag = []
    CB.fly_engine(d, dec, T=0.3, diag=diag, leg_stance=(W["leg_act"], W["stance_q"]))
    up = [r[8] for r in diag]; om = [np.linalg.norm(r[5:8]) for r in diag]
    print(f"{label}: 0.3s後 z={d.qpos[2]:.2f} 最小up={min(up):+.2f} 最大|ω|={max(om):.0f}", flush=True)
# 条件0: フレッシュ (CB.fly相当)
d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[2] = 12; d.qpos[3:7] = W["Q0"]; mujoco.mj_forward(m, d)
engine_spin(d, "フレッシュ")
# 条件1: 立位→歩行なし→テレポート (脚は立位姿勢)
d = mujoco.MjData(m); WB.reset_standing(m, d); WB.settle(m, d, W, 0.5)
d.qpos[2] = 12; d.qpos[3:7] = np.asarray(W["Q0"], float); d.qvel[:] = 0; mujoco.mj_forward(m, d)
engine_spin(d, "立位からテレポート")
# 条件2: 立位→テレポート + act リセット
d = mujoco.MjData(m); WB.reset_standing(m, d); WB.settle(m, d, W, 0.5)
d.qpos[2] = 12; d.qpos[3:7] = np.asarray(W["Q0"], float); d.qvel[:] = 0
if d.act.size: d.act[:] = 0
d.ctrl[:] = 0; mujoco.mj_forward(m, d)
engine_spin(d, "立位→テレポート+act/ctrlリセット")
# 条件3: 立位→テレポート + 全関節をキーフレームへ (体位置・姿勢以外)
d = mujoco.MjData(m); WB.reset_standing(m, d); WB.settle(m, d, W, 0.5)
d0 = mujoco.MjData(m); mujoco.mj_resetData(m, d0)
d.qpos[7:] = d0.qpos[7:]; d.qvel[:] = 0; d.qpos[2] = 12; d.qpos[3:7] = np.asarray(W["Q0"], float)
if d.act.size: d.act[:] = 0
d.ctrl[:] = 0; mujoco.mj_forward(m, d)
engine_spin(d, "立位→全関節リセット+テレポート")
