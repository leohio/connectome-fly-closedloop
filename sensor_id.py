#!/usr/bin/env python
"""統合36d: 実回路の復号誤差の構造を実測する (センサ同定)。

訓練用の回路模型 (定常ガウスノイズ) では実回路への転移が部分的だった。
ES-K での実回路飛行中に (復号ω, 真の遅いω) を記録し、残差の
分散・自己相関時間・|ω|依存性を測って、構造を写した模型を作る材料にする。
"""
import numpy as np
from multiprocessing import Pool


def job(pert):
    import connectome_bioflight as CB
    CB.OMLOG = []
    dec = CB.calibrate()
    r = CB.fly(src="circuit", dec=dec, T=10.0, pert_seed=pert)
    log = CB.OMLOG
    t = np.array([x[0] for x in log])
    est = np.array([x[1] for x in log])
    tru = np.array([x[2] for x in log])
    return dict(pert=pert, srv=r[0], t=t, est=est, tru=tru)


if __name__ == "__main__":
    with Pool(2) as p:
        out = p.map(job, [0, 1])
    for d in out:
        m = d["t"] > 0.15
        e = d["est"][m] - d["tru"][m]
        tru = d["tru"][m]
        dt = np.median(np.diff(d["t"][m]))
        print(f"外乱{d['pert']} (生存{d['srv']:.2f}s, {m.sum()}点, "
              f"dt={dt*1000:.2f}ms):", flush=True)
        for ax, nm in enumerate("xyz"):
            r = e[:, ax]
            ac = np.corrcoef(r[:-1], r[1:])[0, 1]
            tau = -dt / np.log(max(ac, 1e-3))
            cor = np.corrcoef(np.abs(tru[:, ax]), np.abs(r))[0, 1]
            print(f"  {nm}: 残差std={r.std():6.3f} 平均={r.mean():+6.3f} "
                  f"AR1={ac:.3f} (τ≈{tau*1000:.0f}ms) "
                  f"|ω|依存corr={cor:+.2f}", flush=True)
        gain = [np.polyfit(tru[:, ax], d["est"][m][:, ax], 1)[0]
                for ax in range(3)]
        print(f"  実効ゲイン est/tru = {[round(g,2) for g in gain]}",
              flush=True)
    np.savez("outputs/sensor_id.npz",
             **{f"e{d['pert']}_{k}": d[k] for d in out
                for k in ("t", "est", "tru")})
    print("保存 outputs/sensor_id.npz", flush=True)
    print("DONE", flush=True)
