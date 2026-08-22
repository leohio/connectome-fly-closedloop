"""統合36e: 学習Kが実回路で墜落する瞬間の復号を解剖する。"""
import json, numpy as np
from multiprocessing import Pool

def job(pert):
    import numpy as np, json
    import connectome_bioflight as CB
    res = json.load(open("outputs/reward_K_v3_model1.json"))
    r = [x for x in res if x["seed"] == 7][0]
    th = np.array(r["theta"])
    CB.K_POL = th[:CB.N_U * CB.N_X].reshape(CB.N_U, CB.N_X)
    CB.B_POL = th[CB.N_U * CB.N_X:]
    CB.OMLOG = []
    dec = CB.calibrate()
    out = CB.fly(src="circuit", dec=dec, T=10.0, pert_seed=pert)
    log = CB.OMLOG
    t = np.array([x[0] for x in log])
    est = np.array([x[1] for x in log])
    tru = np.array([x[2] for x in log])
    np.savez(f"outputs/autopsy_p{pert}.npz", t=t, est=est, tru=tru,
             srv=out[0])
    return pert, out[0]

if __name__ == "__main__":
    with Pool(2) as p:
        rr = p.map(job, [0, 2])
    for pert, srv in rr:
        z = np.load(f"outputs/autopsy_p{pert}.npz")
        t, est, tru = z["t"], z["est"], z["tru"]
        srv = float(z["srv"])
        print(f"外乱{pert}: 生存{srv:.2f}s", flush=True)
        for t0, t1, lab in [(0.15, srv - 0.5, "巡航"), (max(srv-0.5,0.15), srv, "最後0.5s")]:
            m = (t > t0) & (t <= t1)
            if m.sum() < 10:
                continue
            e = est[m] - tru[m]
            frozen = np.mean(np.all(np.diff(est[m], axis=0) == 0, axis=1))
            print(f"  {lab}: 残差std={np.linalg.norm(e.std(0)):.2f} "
                  f"|真ω|平均={np.linalg.norm(tru[m], axis=1).mean():.1f} "
                  f"|復号ω|平均={np.linalg.norm(est[m], axis=1).mean():.1f} "
                  f"凍結率={frozen:.2f}", flush=True)
            # 符号: 最後に復号が真と逆を向いた割合 (軸ごと)
            sgn = np.mean(np.sign(est[m]) != np.sign(tru[m]), axis=0)
            print(f"    符号不一致率 xyz = {[round(x,2) for x in sgn]}", flush=True)
    print("DONE", flush=True)
