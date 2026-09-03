"""P2c 補遺: 単収縮の壁は「指令更新の位相」の問題か

仮説: 1 羽ばたき 1 回の階段状指令が翅振幅 amp = 1+u[0] に入るとき、更新が位相 0 (sin=0) なら
翅位置は連続で無害、位相がずれると翅位置が不連続になり高周波を注入する。
統合31 の bioflight.py は timestep が周期を割り切らず更新位相が漂う → 瞬時筋では破綻、
単収縮 τ=4.25 ms がそれを平滑化して救った、という説明の直接検証。

条件: 更新位相 off ∈ {0, 0.25, 0.5} 周期 × 筋 {瞬時, 4.25 ms}、ES-K、d=0、3 s。
"""
import sys, os, json
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase_algebra.closed_loop as CL
import phase_algebra.hover_growth as HG
import phase_algebra.ident_plant as IP

GS, A_S, XB_REF = HG.GS, HG.A_S, HG.XB_REF

def run(K, b, d, tau_tw, off_frac, T=3.0, pert=0.02, snap=None):
    d_ = IP.restore(snap)
    IP.perturb(d_, np.array([pert, pert, 0, 0, 0, 0, 0, 0, 0]))
    u_cmd = snap["u"].copy(); u = snap["u"].copy()
    om_f = GS * XB_REF[2:5]; hist = [om_f.copy() for _ in range(d + 1)]
    t0 = d_.time; xb_log = []; acc = np.zeros(9); buf = np.zeros((IP.STEPS, 3)); bi = 0; ssum = np.zeros(3)
    off = int(round(off_frac * IP.STEPS))
    for k in range(int(T / IP.PER) * IP.STEPS):
        t = t0 + k * IP.DTP
        ssum += d_.qvel[3:6] - buf[bi]; buf[bi] = d_.qvel[3:6].copy(); bi = (bi + 1) % IP.STEPS
        if k % IP.STEPS == 0 and k > 0:
            xb_log.append(acc / IP.STEPS); acc[:] = 0
        if k % IP.STEPS == off and k > 0:            # 指令更新を位相 off で行う
            om_bar = ssum / IP.STEPS
            om_f = A_S * om_f + (1 - A_S) * GS * om_bar
            hist.insert(0, om_f.copy()); hist = hist[:d + 1]
            om_hat = hist[d]
            x = IP.state_of(d_)
            xin = np.array([(IP.Z_T - x[5]) / 5.0, -x[6] / 30.0, x[0], x[1], om_hat[0] / 20, om_hat[1] / 20, om_hat[2] / 20])
            u_cmd = np.clip(IP.U_TRIM + np.tanh(K @ xin + b) * 0.35, -0.55, 0.55)
        if tau_tw <= IP.DTP: u = u_cmd.copy()
        else: u += IP.DTP * (u_cmd - u) / tau_tw
        IP.wing_ctrl(d_, u, t)
        mujoco.mj_step(IP.m, d_)
        acc += IP.state_of(d_)
        if not np.isfinite(d_.qpos[2]) or d_.qpos[2] < 0.5 or d_.qpos[2] > 40:
            break
    xb = np.array(xb_log) - XB_REF
    return xb, (k + 1) * IP.DTP

snap, _ = IP.settle(1.0)
K, b = CL.load_K("outputs/bioflight_best.npy")
res = {}
print("更新位相  筋       生存s  成長率/羽ばたき  |e_b| 最終")
for off in (0.0, 0.25, 0.5):
    for lab, tau in (("瞬時", 0.0), ("4.25ms", 4.25e-3)):
        xb, srv = run(K, b, 0, tau, off, snap=snap)
        lam, n = HG.growth_rate(xb)
        eb = float(np.linalg.norm(xb[-1, :2])) if len(xb) else float("nan")
        res[f"off{off}_{lab}"] = dict(off=off, tau=tau, survival=float(srv), lam=(float(lam) if np.isfinite(lam) else None), eb_end=eb)
        print(f"{off:5.2f}     {lab:7s}  {srv:5.2f}  {lam:8.4f}         {eb:.3f}", flush=True)
json.dump(res, open("phase_algebra/outputs/twitch_phase.json", "w"), indent=1, ensure_ascii=False)
print("保存: phase_algebra/outputs/twitch_phase.json")
