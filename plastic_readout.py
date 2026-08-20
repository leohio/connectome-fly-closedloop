#!/usr/bin/env python
"""統合35: 外部較正を捨て、回路自身が局所学習則で読出しを獲得する。

統合34で確定したこと: 行動レベルの配線特異性は「配線と、その配線用に学習した
読出しの適合性」としてのみ存在し、読出しを再較正すれば多くのランダム配線も飛ぶ。
残る本丸は「読出しの獲得過程そのものを生物学にすること」である。

ここで捨てるもの: measure_S の最小二乗較正 (既知のωで探査し pinv を取る外部手続き)。
置き換えるもの: 動物が実際に持つ教師信号による局所デルタ則。
  - 教師 = 視覚由来の遅く不正確なω推定 (ノイズつき)。ハルテアより遅く粗いが曖昧さがない
  - 学習 = W += eta * outer(dphi, err)。各シナプスが前段活動 dphi_i と
    後段の誤差信号 err_ax だけを使う局所則 (小脳様の誤差教師)
  - 汎化試験 = 学習で見せていない、より速い回転。視覚が追えない領域であり、
    ハルテア読出しが存在する生物学的理由そのもの

事前予測 (統合33の実測から): 実配線の条件数2.33に対し full-null は9.24±14.4。
勾配型の学習は条件数が小さいほど速く収束するため、**実配線は学習が速い**はずである。
これが出れば「配線の個性が、行動ではなく学習効率に転写される」ことになる。
"""
import json
import os
import sys
import numpy as np
from multiprocessing import Pool
from brian2 import ms as _ms, prefs
prefs.codegen.target = "numpy"
import phase_reflex as PR
import connectome_fastloop as CF
import locked_circuit as LC
import measure_S as MS

T_SETTLE = 0.12          # 位相が新しいωへ再固定するまで
T_MEAS = 0.15            # 位相推定の窓
N_TRAIN = 120            # 経験する回転エピソード数
W_TRAIN = 8.0            # 訓練中の回転の大きさ (視覚が追える範囲)
W_TEST = 12.0            # 汎化試験の回転 (未見・より速い)
TEACH_REL = 0.15         # 視覚教師の相対誤差
TEACH_ABS = 0.3          # 視覚教師の絶対誤差 [rad/s]
ETA = 0.35               # 学習率 (正規化デルタ則)
CHECK = [10, 20, 40, 80, 120]   # 学習曲線を測る時点


def _phase_of(tr, st_idx, names, t0, t1):
    """指定窓のスパイクから各筋の活動位相を出す"""
    out = {}
    for mu in names:
        st = tr[mu]
        st = st[(st > t0) & (st <= t1)]
        if len(st) < 3:
            continue
        z = np.mean(np.exp(2j * np.pi * np.mod(st * PR.WBF, 1.0)))
        if abs(z) < 0.3:
            continue
        out[mu] = np.angle(z) / (2 * np.pi)
    return out


def episodes(shuffle=None, seed=0, n_train=N_TRAIN, rng_seed=0):
    """1つの配線で、回転エピソード列を経験させ (dphi, omega) を集める。

    ネットワークは1回だけ構築し、ωを切り替えながら連続実行する
    (measure_S のように毎回組み直すと配線読込が支配的になるため)。
    """
    kw = dict(LC.SETUP_KW)
    if shuffle:
        kw["shuffle"] = shuffle
        kw["shuffle_seed"] = seed
    net, mon, pg, pref, side, st_idx, n = PR.setup(**kw)
    names = [mu for mu in MS.MUS if mu in st_idx]
    rg = np.random.default_rng(9000 + rng_seed)
    # 訓練は視覚が追える範囲、試験は未見のより速い回転
    train_om = [rg.normal(0, 1, 3) for _ in range(n_train)]
    train_om = [W_TRAIN * o / max(np.linalg.norm(o), 1e-9)
                * rg.uniform(0.35, 1.0) for o in train_om]
    test_om = [np.array(o, float) for o in
               [(W_TEST, 0, 0), (0, W_TEST, 0), (0, 0, W_TEST),
                (0.7 * W_TEST, -0.7 * W_TEST, 0.4 * W_TEST),
                (-0.6 * W_TEST, 0.5 * W_TEST, -0.8 * W_TEST),
                (0.45 * W_TEST, 0.45 * W_TEST, 0.45 * W_TEST)]]
    seq = [np.zeros(3)] + train_om + test_om
    shift_prev = np.zeros(len(pref))
    t_now = 0.0
    marks = []
    for om in seq:
        o = np.clip(om, -14, 14)
        shift = PR.C_PHASE * (side * o[0] + o[1]
                              + side * np.cos(2 * np.pi * pref) * o[2])
        pg.v = pg.v - (shift - shift_prev)
        shift_prev = shift
        net.run((T_SETTLE + T_MEAS) * 1000 * _ms, namespace={})
        marks.append((t_now + T_SETTLE, t_now + T_SETTLE + T_MEAS))
        t_now += T_SETTLE + T_MEAS
    raw = mon.spike_trains()
    tr = {mu: np.array(raw[st_idx[mu]] / _ms) / 1000.0 for mu in names}
    ph = [_phase_of(tr, st_idx, names, a, b) for a, b in marks]
    base = ph[0]
    good = [mu for mu in names if mu in base
            and all(mu in p for p in ph[1:])]
    if len(good) < 3:
        return None
    def dphi(p):
        return np.array([(p[mu] - base[mu] + 0.5) % 1.0 - 0.5 for mu in good])
    X_tr = np.array([dphi(p) for p in ph[1:1 + n_train]])
    Y_tr = np.array(train_om)
    # 視覚由来の粗い教師。デルタ則と外部最小二乗の比較を公平にするため、
    # ここで一度だけ生成して両者に同じものを渡す (以前は最小二乗にだけ
    # ノイズなしの真値を渡しており、比較として不当だった)
    rgt = np.random.default_rng(7700 + rng_seed)
    Y_teach = np.array([y + rgt.normal(
        0, TEACH_REL * np.linalg.norm(y) + TEACH_ABS, 3) for y in Y_tr])
    X_te = np.array([dphi(p) for p in ph[1 + n_train:]])
    Y_te = np.array(test_om)
    return dict(X_tr=X_tr, Y_tr=Y_tr, Y_teach=Y_teach, X_te=X_te, Y_te=Y_te,
                n_mus=len(good), names=good,
                phi0=[float(base[mu]) for mu in good])


def learn(d, rng_seed=0, checkpoints=CHECK):
    """局所デルタ則で読出しWを獲得する。教師は視覚由来の粗いω。"""
    X, T = d["X_tr"], d["Y_teach"]
    W = np.zeros((X.shape[1], 3))
    curve = {}
    for i in range(len(X)):
        x, teach = X[i], T[i]
        err = teach - x @ W
        nx = float(x @ x) + 1e-12
        W = W + ETA * np.outer(x, err) / nx     # 正規化デルタ則 (局所)
        if (i + 1) in checkpoints:
            curve[i + 1] = _test_err(W, d)
    return W, curve


def _test_err(W, d):
    P = d["X_te"] @ W
    return float(np.mean([np.linalg.norm(P[k] - d["Y_te"][k])
                          / max(np.linalg.norm(d["Y_te"][k]), 1e-9)
                          for k in range(len(P))]))


def lsq_err(d):
    """比較用: 外部最小二乗 (統合33までのやり方) を同じ訓練データで解いた場合"""
    W, *_ = np.linalg.lstsq(d["X_tr"], d["Y_teach"], rcond=None)
    return _test_err(W, d)


N_DRAW = 3   # 全構成に同数の教師・エピソード系列を与えて平均する
              # (実配線だけ複数系列にすると順位検定の交換可能性が壊れるため)


def job(a):
    tag, seed = a
    sh = "all" if tag == "shuffle" else None
    try:
        curves, lsqs, conds, n_mus = [], [], [], None
        for dr in range(N_DRAW):
            d = episodes(shuffle=sh, seed=seed, rng_seed=1000 * dr + seed)
            if d is None:
                return dict(tag=tag, seed=seed, failed="位相固定筋が3本未満")
            W, curve = learn(d, rng_seed=1000 * dr + seed)
            curves.append(curve)
            lsqs.append(lsq_err(d))
            conds.append(float(np.linalg.cond(d["X_tr"])))
            n_mus = d["n_mus"]
        curve = {k: float(np.mean([c[k] for c in curves]))
                 for k in curves[0]}
        return dict(tag=tag, seed=seed, n_mus=n_mus, n_draw=N_DRAW,
                    curve={str(k): v for k, v in curve.items()},
                    final=curve[max(curve)], lsq=float(np.mean(lsqs)),
                    cond_X=float(np.mean(conds)))
    except Exception as exc:
        return dict(tag=tag, seed=seed, failed=str(exc))


if __name__ == "__main__":
    n_sh = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    jobs = [("real", 0)] + [("shuffle", s) for s in range(n_sh)]
    with Pool(min(len(jobs), 13)) as p:
        res = p.map(job, jobs)
    real = [r for r in res if r["tag"] == "real"][0]
    sh = [r for r in res if r["tag"] == "shuffle" and not r.get("failed")]
    nf = len([r for r in res if r["tag"] == "shuffle" and r.get("failed")])
    if real.get("failed"):
        print("実配線で失敗:", real["failed"], flush=True)
        sys.exit(1)
    print(f"局所デルタ則による読出し獲得 (訓練{N_TRAIN}エピソード, "
          f"教師=視覚由来の粗いω, 試験=未見の速い回転{W_TEST}rad/s)", flush=True)
    print(f"  n=full-null {len(sh)}構成" + (f" (較正失敗{nf})" if nf else ""),
          flush=True)
    print("\n学習曲線 (未見回転での相対復号誤差):", flush=True)
    print(f"{'エピソード':>10s} {'実配線':>9s} {'ヌル中央値':>11s} "
          f"{'ヌル平均':>9s} {'実以下':>8s} {'片側p':>8s}", flush=True)
    for c in CHECK:
        rv = real["curve"][str(c)]
        v = np.array([r["curve"][str(c)] for r in sh], float)
        v = v[np.isfinite(v)]
        nb = int((v <= rv).sum())
        print(f"{c:>10d} {rv:>9.4f} {np.median(v):>11.4f} {v.mean():>9.4f} "
              f"{nb:>4d}/{len(v):<3d} {(nb + 1) / (len(v) + 1):>8.4f}",
              flush=True)
    lv = np.array([r["lsq"] for r in sh], float)
    lv = lv[np.isfinite(lv)]
    nb = int((lv <= real["lsq"]).sum())
    print(f"\n参考: 同じ訓練データを外部最小二乗で解いた場合", flush=True)
    print(f"  実配線={real['lsq']:.4f} ヌル中央値={np.median(lv):.4f} "
          f"実以下={nb}/{len(lv)} p={(nb + 1) / (len(lv) + 1):.4f}", flush=True)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/plastic_readout.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("保存 outputs/plastic_readout.json", flush=True)
    print("DONE", flush=True)
