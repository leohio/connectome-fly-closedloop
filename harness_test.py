import sys, numpy as np
import walk_base as WB          # 床接触ON (シーケンスと同じモデル状態)
import locked_circuit as LC
import connectome_bioflight as CB
cond = sys.argv[1]
W = WB.env()
if "legs" in cond:
    CB.LEG_STANCE = (W["leg_act"], W["stance_q"])
if "tau5" in cond:
    LC.TAU_P = 0.005
dec = CB.calibrate()
r = CB.fly(src="circuit", dec=dec, T=4.0, pert_seed=0)
print(f"{cond}: 生存{r[0]:.2f}s 直立{r[1]:+.2f} 高度誤差{r[2]:.1f}", flush=True)
