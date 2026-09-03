"""P2c: 筋単収縮の必要性は固有値問題から出るか

実験ログ (統合31): 単収縮 (τ=4.25 ms) が無いと「1 羽ばたき 1 回の階段状指令が高周波を注入し、
全候補が 0.58 s 以下に崩れる」。入れた途端に 3.00 s に到達。

検証: (a) 線形 — closed_loop.build_T の a_tw, b_tw を τ_tw の関数として変え、速い姿勢モードの |λ| を見る。
      (b) 非線形 — hover_growth.run で IP.TAU_TW を変えて ES-K の生存時間と成長率を測る。
(a) は羽ばたき平均の遅い成分しか見ないので、「高周波注入」が壁なら (a) では壁が出ず (b) で出るはず。
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase_algebra.closed_loop as CL
import phase_algebra.hover_growth as HG
import phase_algebra.ident_plant as IP

PER, DTP = IP.PER, IP.DTP
TAUS = [("瞬時", DTP), ("1ms", 1e-3), ("2ms", 2e-3), ("4.25ms", 4.25e-3), ("8ms", 8e-3), ("15ms", 15e-3)]

def set_tw(tau):
    CL.A_TW = float(np.exp(-PER / tau))
    CL.B_TW = float((tau / PER) * (1 - np.exp(-PER / tau)))

res = {"linear": {}, "nonlinear": {}}
print("=== (a) 線形予測: 速い姿勢モード |λ| ===")
for name, src in [("ES-K", "outputs/bioflight_best.npy"), ("v6", "outputs/reward_K_v6_delay2.json:2")]:
    K, b = CL.load_K(src)
    print(f"{name}   " + "  ".join(f"d{d}" for d in range(4)))
    res["linear"][name] = {}
    for lab, tau in TAUS:
        set_tw(tau)
        row = [CL.analyse(CL.build_T(K, b, d, False, "circuit"))[1] for d in range(4)]
        res["linear"][name][lab] = row
        print(f"  τ={lab:7s} " + "  ".join(f"{v:.4f}" for v in row), flush=True)
set_tw(4.25e-3)

print("\n=== (b) 非線形実走行 ES-K (3 s、摂動 0.02 rad) ===")
snap, _ = IP.settle(1.0)
K, b = CL.load_K("outputs/bioflight_best.npy")
print("τ_tw     d   生存s   実測成長率/羽ばたき")
for lab, tau in TAUS:
    IP.TAU_TW = tau
    res["nonlinear"][lab] = {}
    for d in (0, 1):
        xb, srv = HG.run(K, b, d, T=3.0, snap=snap)
        lam, n = HG.growth_rate(xb)
        res["nonlinear"][lab][f"d{d}"] = dict(survival=float(srv), lam=float(lam) if np.isfinite(lam) else None, n_lin=int(n))
        print(f"{lab:7s}  {d}   {srv:5.2f}   {lam:.4f} (線形域 {n})", flush=True)
IP.TAU_TW = 4.25e-3
json.dump(res, open("phase_algebra/outputs/twitch_wall.json", "w"), indent=1, ensure_ascii=False)
print("保存: phase_algebra/outputs/twitch_wall.json")
