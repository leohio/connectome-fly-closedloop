#!/usr/bin/env python
"""P2 厳密検証: 摂動あり/なしの2軌道の差から小摂動の成長率 (Floquet乗数) を測る。

ホバーは固定点ではなく振幅~0.1 rad の持続振動 (リミットサイクル) なので、
|δe_b| の包絡や生存時間では線形成長率を測れない。同一条件で摂動 0.02 rad を
与えた軌道と与えない軌道の差 Δ[k] を取り、線形域 (|Δ|<0.3) で log|Δ| を直線フィットする。
得られる exp(傾き) が実測の |λ| (1羽ばたきあたり) であり、閉ループ行列 T の
速い姿勢モードの予測と直接比較できる。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.ident_plant as IP
import phase_algebra.hover_growth as HG
import phase_algebra.closed_loop as CL
from phase_algebra.closed_loop import load_K, build_T, analyse


def floquet_emp(K, b, d, snap, T=2.5):
    base, _ = HG.run(K, b, d, T=T, pert=0.0, snap=snap)
    pert, srv = HG.run(K, b, d, T=T, pert=0.02, snap=snap)
    n = min(len(base), len(pert))
    D = pert[:n] - base[:n]
    a = np.linalg.norm(D[:, :5] / np.array([1, 1, 20, 20, 20]), axis=1)   # 姿勢+角速度(正規化)
    ok = np.where(a < 0.3)[0]
    i1 = int(ok[-1]) if len(ok) else 0
    # 速いモード (周期~18羽ばたき) は初期窓で減衰/成長する。長窓で当てはめると
    # 減衰後に残る中立な位置ドリフト (|λ|=1) を測ってしまうため、初期窓 5〜60 羽ばたきの
    # 振動包絡 (20羽ばたき窓の最大) を用いる
    env = np.array([a[max(0, i - 10):i + 11].max() for i in range(len(a))])
    i0, i_end = 5, min(60, i1)
    if i_end - i0 < 8:
        return np.nan, srv, i1
    sl = np.polyfit(np.arange(i0, i_end), np.log(env[i0:i_end] + 1e-12), 1)[0]
    return float(np.exp(sl)), srv, i1


if __name__ == "__main__":
    ctrls = [("ES-K (bioflight_best)", "outputs/bioflight_best.npy"),
             ("ES 遅延2訓練 (es_takeoff_best)", "outputs/es_takeoff_best.npy")]
    out = {}
    for name, src in ctrls:
        K, b = load_K(src)
        IP.K_POL[:] = K; IP.B_POL[:] = b
        snap, _ = IP.settle(1.5)                       # 各制御器を自分の動作点で静定
        xb_own = IP.trajectory(snap, np.zeros(9), np.tile(snap["u"], (2, 1)))[0]
        CL.XB_REF = xb_own
        print(f"\n{name}")
        print("  d   |λ|_emp(初期窓の包絡)  |λ|_pred(速いモード)  生存s  線形域(羽ばたき)")
        rows = []
        for d in range(0, 5):
            lam_e, srv, n_lin = floquet_emp(K, b, d, snap)
            _, lam_p, per_ms, _ = analyse(build_T(K, b, d, False, "circuit"))
            print(f"  {d}      {lam_e:7.4f}            {lam_p:7.4f}          {srv:5.2f}   {n_lin}", flush=True)
            rows.append(dict(d=d, lam_emp=float(lam_e), lam_pred=float(lam_p), survival=float(srv), n_lin=int(n_lin)))
        out[name] = rows
    json.dump(out, open("phase_algebra/outputs/floquet_check.json", "w"), indent=1, ensure_ascii=False)
    print("\n保存: phase_algebra/outputs/floquet_check.json")
