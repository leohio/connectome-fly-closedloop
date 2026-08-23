import sys, numpy as np
import reward_K as RK
RK.SENSOR = "circuit_model"; RK.FREQ_REFLEX = 0.0
ramp, jump = float(sys.argv[1]), float(sys.argv[2])
RK.JUMP_VZ = jump
th_es = np.load("outputs/bioflight_best.npy")
Z0, ZC = 1.5, 12.0
def zfn(t):
    tf = t - 0.3
    if tf < 0: return Z0
    if tf < ramp: return Z0 + tf / ramp * (ZC - Z0)
    return ZC
RK.DEBUG_LOG = []
o = RK.rollout_sensed(th_es, None, T=7.0, z_fn=zfn, hold_until=0.3,
                      z_floor=0.05, start_z=Z0, clamp_x0=False)
zl = [r[1] for r in RK.DEBUG_LOG if r[0] > 5.0]
up = [r[4] for r in RK.DEBUG_LOG if r[0] > 0.3]
print(f"ランプ{ramp}s 押出{jump:.0f}: 生存{o[0]:.2f}s 最小up={min(up):+.2f} 巡航z={np.mean(zl) if zl else 0:.1f}", flush=True)
