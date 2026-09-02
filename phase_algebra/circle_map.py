#!/usr/bin/env python
"""P3 変種E: 円周写像 (位相同期ループ) の構造的代理復号器。

操舵MNは AHP により 0.9 周期の不応期を持ち、その明けた直後の窓に最初に到着する
十分強い求心性入力で発火する。これは 1 次元の円周写像:

    θ_{k+1} = min{ φ_j : φ_j ∈ (θ_k + r, θ_k + 1] (mod 1),  W_ij ≥ w_min }

の不動点であり、どの求心性クラスタ (pref 0 / ½) にロックするかは重みの argmax ではなく
到着順序と不応期で決まる (WTA が弱い予測子だった理由)。ω による位相シフトを入れて
不動点を追跡し、measure_S と同じ手続きで S と未見回転の復号誤差を出す。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import phase_algebra.wiring_operator as WO
import phase_reflex as PR

R_REF = 0.9          # 不応期 [cycle] (setup: neu.rfc = 0.9/WBF)
W_MIN = 4.0          # 発火を起こせる入力の最小重み [mV] (電気 8mV は必ず通る)


def lock(phis, theta0, iters=60):
    """円周写像を反復して不動点 (ロック位相) を返す。phis: 到着位相 (mod 1)"""
    th = theta0
    for _ in range(iters):
        gap = (phis - (th + R_REF)) % 1.0          # 窓の開始からの距離
        th_new = (th + R_REF + gap.min()) % 1.0
        if abs(((th_new - th + 0.5) % 1.0) - 0.5) < 1e-9:
            return th_new
        th = th_new
    return th


def mn_phase(W, pref, side, om, theta_prev=None):
    o = np.clip(np.asarray(om, float), -12, 12)
    shift = PR.C_PHASE * (side * o[0] + o[1] + side * np.cos(2 * np.pi * pref) * o[2])
    phi = np.mod(pref + shift, 1.0)
    out = np.full(len(W), np.nan)
    for i in range(len(W)):
        js = np.where(W[i] >= W_MIN)[0]
        if len(js) == 0: continue
        th0 = theta_prev[i] if theta_prev is not None and np.isfinite(theta_prev[i]) else np.angle(np.sum(W[i, js] * np.exp(2j * np.pi * phi[js]))) / (2 * np.pi) % 1.0
        out[i] = lock(phi[js], th0)
    return out


def surrogate(W, pref, side):
    base = mn_phase(W, pref, side, (0, 0, 0))
    good = np.isfinite(base)
    S = np.zeros((len(W), 3))
    for ax in range(3):
        op = [0.0] * 3; op[ax] = WO.W_CAL; om_ = [0.0] * 3; om_[ax] = -WO.W_CAL
        pp = mn_phase(W, pref, side, op, base); pm = mn_phase(W, pref, side, om_, base)
        S[:, ax] = (((pp - base + 0.5) % 1 - 0.5) - ((pm - base + 0.5) % 1 - 0.5)) / (2 * WO.W_CAL)
    good &= np.isfinite(S).all(1)
    if good.sum() < 3: return dict(err=np.nan, cond=np.nan, n_good=int(good.sum()))
    Dec = np.linalg.pinv(S[good]).T
    errs = []
    for om in WO.TESTS:
        p = mn_phase(W, pref, side, om, base)
        dphi = (p[good] - base[good] + 0.5) % 1 - 0.5
        errs.append(np.linalg.norm(dphi @ Dec - np.array(om)) / np.linalg.norm(om))
    sv = np.linalg.svd(S[good], compute_uv=False)
    return dict(err=float(np.mean(errs)), cond=float(sv[0] / max(sv[-1], 1e-12)), n_good=int(good.sum()),
                weak=[float(v) for v in np.linalg.svd(S[good])[2][-1]])


def main():
    dq = {(r["tag"], r["seed"]): r for r in json.load(open("outputs/decode_quality_full_null.json"))}
    rows = []
    for tag, seed in [("real", 0)] + [("shuffle", s) for s in range(100)]:
        ex = WO.extract("all" if tag == "shuffle" else None, seed)
        W = np.clip(WO.effective_W(ex, hops=0), 0, None)
        r = surrogate(W, ex["pref"], ex["side"]); r.update(tag=tag, seed=seed)
        m = dq.get((tag, seed), {}); r["cond_meas"] = m.get("cond", np.nan); r["err_meas"] = m.get("err", np.nan)
        rows.append(r)
    json.dump(rows, open("phase_algebra/outputs/circle_map.json", "w"), indent=1)
    real = rows[0]; sh = [r for r in rows[1:] if np.isfinite(r["err"]) and np.isfinite(r["err_meas"])]
    def rank_(v): o = np.argsort(v); rr = np.empty(len(v)); rr[o] = np.arange(len(v)); return rr
    xe = np.log(np.array([r["err"] for r in sh]) + 1e-9); xc = np.log(np.array([min(r["cond"], 1e6) for r in sh]))
    ym = np.log(np.array([r["cond_meas"] for r in sh])); ye = np.log(np.array([r["err_meas"] for r in sh]))
    print(f"円周写像代理 (ヌル n={len(sh)}):")
    print(f"  実配線: 代理誤差 {real['err']:.3f} / 実測 {real['err_meas']:.3f};  代理cond {real['cond']:.2f} / 実測 {real['cond_meas']:.2f}")
    print(f"  ヌル中央値: 代理誤差 {np.median(np.exp(xe)):.3f} / 実測 {np.median(np.exp(ye)):.3f};  代理cond {np.median(np.exp(xc)):.2f} / 実測 {np.median(np.exp(ym)):.2f}")
    print(f"  代理誤差 → 実測誤差: Spearman ρ={np.corrcoef(rank_(xe), rank_(ye))[0,1]:+.3f}")
    print(f"  代理cond → 実測cond: Spearman ρ={np.corrcoef(rank_(xc), rank_(ym))[0,1]:+.3f}")
    print(f"  代理cond → 実測誤差: Spearman ρ={np.corrcoef(rank_(xc), rank_(ye))[0,1]:+.3f}")
    print(f"  実配線の代理誤差はヌル{len(sh)}個中 {(np.exp(xe) <= real['err']).sum()} 個より悪い/同等")


if __name__ == "__main__":
    main()
