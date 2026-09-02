#!/usr/bin/env python
"""P3: 配線特異性の行列不変量 — 複素位相和モデル (構造的代理復号器)。

回路は羽ばたき位相固定で動く (第1層)。求心性 j は好み位相 pref_j に体回転 ω による
シフト shift_j(ω) = C_PHASE·(side_j ω_x + ω_y + side_j cos(2π pref_j) ω_z) を加えた位相で
1周期1発撃つ。操舵MN i への入力を複素フェーザ和で書く:

    Z_i(ω) = Σ_j W_ij · exp( 2πi (pref_j + shift_j(ω)) )       θ_i(ω) = arg Z_i / 2π

W は配線 (直接化学 + 補完電気、任意で介在1ホップ) から取る。これはスパイキング計算を
含まない純粋な行列計算であり、measure_S と同じ手続き (±10 で較正、未見回転で復号) を
この代理モデルに適用すれば、実配線と 100 ヌルの「未見回転の復号誤差」を構造だけから
予測できるかを検定できる。線形性を決める候補不変量は入力位相のコヒーレンス
|Z_i(0)| / Σ_j |W_ij| (複素ベクトルのノルム比)。
"""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from brian2 import Synapses, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import locked_circuit as LC
import measure_S as MS

W_CAL = MS.W_PROBE
TESTS = [(4.0, 0, 0), (0, 4.0, 0), (0, 0, 4.0), (3.0, -3.0, 2.0), (-6.0, 2.0, -4.0)]   # decode_quality と同一


def extract(shuffle=None, seed=0):
    """PR.setup で組んだ回路から配線配列・符号化・MN指標を取り出す (Brian2は走らせない)。
    結果は phase_algebra/outputs/cache/ に npz でキャッシュ (PR.setup が支配的コストのため)"""
    os.makedirs("phase_algebra/outputs/cache", exist_ok=True)
    cpath = f"phase_algebra/outputs/cache/wiring_{shuffle or 'real'}_{seed}.npz"
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return {k: (z[k].tolist() if k == "mus" else (int(z[k]) if k in ("n", "n_chem", "n_el") else z[k])) for k in z.files}
    kw = dict(LC.SETUP_KW)
    if shuffle:
        kw["shuffle"] = shuffle; kw["shuffle_seed"] = seed
    net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
    syns = {o.name: o for o in net.objects if isinstance(o, Synapses)}
    chem = syns["ssyn"]
    A_i = np.array(chem.i[:], dtype=int); A_j = np.array(chem.j[:], dtype=int); A_w = np.array(chem.w[:] / PR.mV)
    # 求心性 (pg) → 回路ニューロンの対応
    sh = [s for nm, s in syns.items() if nm not in ("ssyn", "elsyn", "rsyn") and s.source is pg][0]
    h_tgt = np.array(sh.j[:], dtype=int)[np.argsort(np.array(sh.i[:], dtype=int))]
    n_h = len(pref)
    el = syns.get("elsyn")
    E = np.zeros((n, n_h))
    if el is not None:
        for i_, j_ in zip(np.array(el.i[:], dtype=int), np.array(el.j[:], dtype=int)):
            E[j_, i_] += 8.0
    mus = [mu for mu in MS.MUS if mu in st_idx]
    mn = np.array([st_idx[mu] for mu in mus])
    out = dict(n=n, A_i=A_i, A_j=A_j, A_w=A_w, h_tgt=h_tgt, E=E, pref=pref, side=side, mus=np.array(mus), mn=mn,
               n_chem=len(A_i), n_el=int((E > 0).sum()))
    np.savez_compressed(cpath, **out)
    out["mus"] = mus
    return out


def effective_W(ex, hops=1):
    """求心性 j → MN i の実効重み W (n_MN × n_h)。直接化学 + 電気 + (介在1ホップ、符号つき)"""
    n = ex["n"]
    A = np.zeros((n, n)); np.add.at(A, (ex["A_j"], ex["A_i"]), ex["A_w"])   # A[post, pre] (1556² は密で十分)
    W0 = A[:, ex["h_tgt"]]                        # 直接化学 (求心性ニューロンからの出力)
    W = W0.copy()
    if hops >= 1:
        W = W + (A @ W0) * 0.5                    # 介在1ホップ (減衰0.5、符号保持)
    W = W + ex["E"]                               # 電気
    return W[ex["mn"]]


def phases(W, pref, side, om):
    o = np.clip(np.asarray(om, float), -12, 12)
    shift = PR.C_PHASE * (side * o[0] + o[1] + side * np.cos(2 * np.pi * pref) * o[2])
    Z = W @ np.exp(2j * np.pi * (pref + shift))
    return np.angle(Z) / (2 * np.pi), np.abs(Z)


def surrogate_decode(W, pref, side):
    """measure_S と同じ手続きを代理モデルに適用: ±W_CAL で S を較正し、未見回転を復号"""
    base, _ = phases(W, pref, side, (0, 0, 0))
    S = np.zeros((len(W), 3))
    for ax in range(3):
        om_p = [0.0] * 3; om_p[ax] = W_CAL
        om_m = [0.0] * 3; om_m[ax] = -W_CAL
        pp, _ = phases(W, pref, side, om_p); pm, _ = phases(W, pref, side, om_m)
        dp = (pp - base + 0.5) % 1.0 - 0.5; dm = (pm - base + 0.5) % 1.0 - 0.5
        S[:, ax] = (dp - dm) / (2 * W_CAL)
    Dec = np.linalg.pinv(S).T
    errs = []
    for om in TESTS:
        p, _ = phases(W, pref, side, om)
        dphi = (p - base + 0.5) % 1.0 - 0.5
        est = dphi @ Dec
        errs.append(np.linalg.norm(est - np.array(om)) / np.linalg.norm(om))
    sv = np.linalg.svd(S, compute_uv=False)
    coh = np.abs(W @ np.exp(2j * np.pi * pref)) / (np.abs(W).sum(1) + 1e-12)
    return dict(err=float(np.mean(errs)), cond=float(sv[0] / max(sv[-1], 1e-12)), sens=float(np.linalg.norm(S)),
                coh_mean=float(coh.mean()), coh_min=float(coh.min()))


def main(n_null=100, hops=1):
    dq = json.load(open("outputs/decode_quality_full_null.json"))
    meas = {(r["tag"], r["seed"]): r for r in dq}
    rows = []
    for tag, seed in [("real", 0)] + [("shuffle", s) for s in range(n_null)]:
        ex = extract("all" if tag == "shuffle" else None, seed)
        W = effective_W(ex, hops=hops)
        r = surrogate_decode(W, ex["pref"], ex["side"])
        r.update(tag=tag, seed=seed, n_chem=ex["n_chem"], n_el=ex["n_el"], n_mn=len(ex["mn"]))
        m = meas.get((tag, seed))
        r["err_meas"] = m["err"] if m else np.nan
        rows.append(r)
        if tag == "real" or seed < 3 or seed % 20 == 0:
            print(f"{tag}{seed if tag=='shuffle' else ''}: 化学{ex['n_chem']} 電気{ex['n_el']} MN{len(ex['mn'])} "
                  f"代理誤差={r['err']:.3f} 実測誤差={r['err_meas']:.3f} coh={r['coh_mean']:.2f} cond={r['cond']:.1f}", flush=True)
    json.dump(rows, open(f"phase_algebra/outputs/wiring_operator_h{hops}.json", "w"), indent=1)
    sh = [r for r in rows if r["tag"] == "shuffle" and np.isfinite(r["err_meas"])]
    real = rows[0]
    def _rank(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    def pearsonr(a, b):
        r = float(np.corrcoef(a, b)[0, 1]); n_ = len(a)
        t = r * np.sqrt(max(n_ - 2, 1) / max(1 - r * r, 1e-12))
        # 両側 p 値 (t 分布の正規近似で十分)
        from math import erfc, sqrt
        return r, erfc(abs(t) / sqrt(2))
    def spearmanr(a, b):
        return pearsonr(_rank(np.asarray(a)), _rank(np.asarray(b)))
    for key, lab in [("err", "代理復号誤差"), ("coh_mean", "コヒーレンス平均"), ("coh_min", "コヒーレンス最小"), ("cond", "代理条件数")]:
        x = np.array([r[key] for r in sh]); y = np.log(np.array([r["err_meas"] for r in sh]))
        rs, ps = spearmanr(x, y); rp, pp = pearsonr(x if key != "err" else np.log(x + 1e-9), y)
        rank = int((x <= real[key]).sum()) if key in ("err", "cond") else int((x >= real[key]).sum())
        print(f"\n{lab}: Spearman ρ={rs:+.3f} (p={ps:.1e}), Pearson(log) r={rp:+.3f} (p={pp:.1e});"
              f" 実配線の値={real[key]:.3f} はヌル{len(sh)}個中 {rank} 個より良い/同等")
    print(f"\n実測: 実配線誤差 {real['err_meas']:.3f} vs ヌル中央値 {np.median([r['err_meas'] for r in sh]):.3f}")
    print(f"代理: 実配線誤差 {real['err']:.3f} vs ヌル中央値 {np.median([r['err'] for r in sh]):.3f}")
    print(f"\n保存: phase_algebra/outputs/wiring_operator_h{hops}.json")


if __name__ == "__main__":
    hops = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(hops=hops)
