import json, numpy as np
from multiprocessing import Pool
import reward_K as RK

def job(g):
    res = json.load(open("outputs/reward_K_v3_model1.json"))
    r = [x for x in res if x["seed"] == 7][0]
    th = np.concatenate([RK.INNATE, np.array(r["theta"])])
    return RK.rollout_sensed(th, g, sensor="circuit_model",
                             noise_seed=200 + g)[0]

if __name__ == "__main__":
    with Pool(6) as p:
        sv = p.map(job, range(6))
    print("個体7 (旧模型で6/6完走, 実回路2/4墜落) を新模型で:", flush=True)
    print("  生存:", ", ".join(f"{x:.2f}" for x in sv),
          f" 平均{np.mean(sv):.2f}s", flush=True)
    print("DONE", flush=True)
