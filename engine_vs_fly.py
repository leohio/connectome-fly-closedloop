import sys, numpy as np, mujoco
import walk_base as WB
import connectome_bioflight as CB
W = WB.env(); m = W["m"]; dec = CB.calibrate()
cond = sys.argv[1]
if cond == "fly_pert0":
    r = CB.fly(src="circuit", dec=dec, T=3.0, pert_seed=0); print(f"{cond}: 生存{r[0]:.2f} 直立{r[1]:+.2f} 高度誤差{r[2]:.1f}", flush=True)
elif cond == "fly_nopert":
    r = CB.fly(src="circuit", dec=dec, T=3.0, pert_seed=None); print(f"{cond}: 生存{r[0]:.2f} 直立{r[1]:+.2f} 高度誤差{r[2]:.1f}", flush=True)
else:
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d); d.qpos[2] = 12; d.qpos[3:7] = W["Q0"]
    if cond == "engine_pert0": d.qvel[3:6] = np.random.default_rng(0).normal(0, 1.5, 3)
    mujoco.mj_forward(m, d); diag = []
    tf, reason, _ = CB.fly_engine(d, dec, T=3.0, diag=diag)
    up = [r[8] for r in diag]; z = [r[1] for r in diag]
    print(f"{cond}: {reason} {tf:.2f}s 最小up={min(up):+.2f} 平均up={np.mean(up):+.2f} 高度誤差{np.mean(np.abs(np.array(z)-12)):.1f}", flush=True)
