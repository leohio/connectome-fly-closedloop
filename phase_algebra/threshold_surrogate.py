#!/usr/bin/env python
"""P3 変種B: 閾値交差型の構造的代理復号器。

位相和 (arg Σ W e^{iφ}) の代理は実測誤差と無相関だった。実回路の操舵MNは AHP で
1周期1発であり、発火位相は入力の「重心」ではなく、不応期明け後に PSP 和が
**最初に閾値を越える時刻** (順序統計) で決まる。これを構造だけで模す:

    g_i(φ; ω) = Σ_j W_ij · κ((φ − φ_j(ω)) mod 1),   κ(x) = exp(−x·T/τ_PSP)
    θ_i(ω) = min{ φ : g_i(φ; ω) ≥ h_i },  h_i = 0.5 · max_φ g_i(φ; 0)   (較正点で固定)

W は wiring_operator と同じ (直接化学 + 電気、hops で介在1ホップ)。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.wiring_operator as WO
import phase_reflex as PR

NB = 400                       # 周期内の位相ビン
TAU_PSP = 0.005                # PSP 時定数 [s]
T_CYC = 1.0 / PR.WBF
KAP = np.exp(-(np.arange(NB) / NB) * T_CYC / TAU_PSP)     # κ(x), x∈[0,1)


def drive(W, pref, side, om):
    o = np.clip(np.asarray(om, float), -12, 12)
    shift = PR.C_PHASE * (side * o[0] + o[1] + side * np.cos(2 * np.pi * pref) * o[2])
    phi = np.mod(pref + shift, 1.0)
    bins = (phi * NB).astype(int) % NB
    G = np.zeros((len(W), NB))
    for j in range(len(pref)):
        col = W[:, j]
        if not np.any(col): continue
        G += np.outer(col, np.roll(KAP, bins[j]))
    return G


def fire_phase(G, h):
    out = np.full(len(G), np.nan)
    for i in range(len(G)):
        idx = np.where(G[i] >= h[i])[0]
        if len(idx): out[i] = idx[0] / NB
    return out


def surrogate(W, pref, side):
    G0 = drive(W, pref, side, (0, 0, 0)); h = 0.5 * G0.max(1)
    base = fire_phase(G0, h)
    S = np.zeros((len(W), 3))
    for ax in range(3):
        op = [0.0] * 3; op[ax] = WO.W_CAL; om_ = [0.0] * 3; om_[ax] = -WO.W_CAL
        pp = fire_phase(drive(W, pref, side, op), h); pm = fire_phase(drive(W, pref, side, om_), h)
        S[:, ax] = ((pp - base + 0.5) % 1 - 0.5 - ((pm - base + 0.5) % 1 - 0.5)) / (2 * WO.W_CAL)
    good = np.isfinite(S).all(1) & np.isfinite(base)
    if good.sum() < 3: return dict(err=np.nan, n_good=int(good.sum()))
    Dec = np.linalg.pinv(S[good]).T
    errs = []
    for om in WO.TESTS:
        p = fire_phase(drive(W, pref, side, om), h)
        dphi = (p[good] - base[good] + 0.5) % 1 - 0.5
        if not np.isfinite(dphi).all(): errs.append(np.nan); continue
        errs.append(np.linalg.norm(dphi @ Dec - np.array(om)) / np.linalg.norm(om))
    sv = np.linalg.svd(S[good], compute_uv=False)
    return dict(err=float(np.nanmean(errs)), n_good=int(good.sum()), cond=float(sv[0] / max(sv[-1], 1e-12)))


def main(hops=0, n_null=100):
    dq = json.load(open("outputs/decode_quality_full_null.json"))
    meas = {(r["tag"], r["seed"]): r["err"] for r in dq}
    rows = []
    for tag, seed in [("real", 0)] + [("shuffle", s) for s in range(n_null)]:
        ex = WO.extract("all" if tag == "shuffle" else None, seed)
        W = np.clip(WO.effective_W(ex, hops=hops), 0, None)     # 閾値モデルでは興奮性のみ
        r = surrogate(W, ex["pref"], ex["side"]); r.update(tag=tag, seed=seed, err_meas=meas.get((tag, seed), np.nan))
        rows.append(r)
        if tag == "real" or seed < 2: print(f"{tag}{seed}: 代理誤差={r['err']:.3f} 実測={r['err_meas']:.3f} good={r['n_good']}", flush=True)
    json.dump(rows, open(f"phase_algebra/outputs/threshold_surrogate_h{hops}.json", "w"), indent=1)
    sh = [r for r in rows if r["tag"] == "shuffle" and np.isfinite(r["err"]) and np.isfinite(r["err_meas"])]
    def rank(v): o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v)); return rr
    x = np.array([r["err"] for r in sh]); y = np.array([r["err_meas"] for r in sh])
    rs = np.corrcoef(rank(x), rank(y))[0, 1]; rp = np.corrcoef(np.log(x + 1e-9), np.log(y))[0, 1]
    real = rows[0]
    print(f"\n閾値交差代理 (hops={hops}): Spearman ρ={rs:+.3f}, Pearson(log) r={rp:+.3f}  (n={len(sh)})")
    print(f"  実配線: 代理{real['err']:.3f} / 実測{real['err_meas']:.3f};  ヌル中央値: 代理{np.median(x):.3f} / 実測{np.median(y):.3f}")
    print(f"  実配線の代理誤差はヌル{len(sh)}個中 {(x <= real['err']).sum()} 個より悪い")


if __name__ == "__main__":
    main(hops=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
