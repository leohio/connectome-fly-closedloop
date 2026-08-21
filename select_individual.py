import json, numpy as np
from multiprocessing import Pool
import reward_K as RK

def job(a):
    seed, gust = a
    res = json.load(open("outputs/reward_K.json"))
    r = [x for x in res if x["seed"] == seed][0]
    th = np.concatenate([RK.INNATE, np.array(r["theta"])])
    return (seed, RK.rollout_sensed(th, gust, sensor="circuit_model",
                                    noise_seed=200 + gust)[0])

if __name__ == "__main__":
    jobs = [(s, g) for s in range(8) for g in range(6)]
    with Pool(16) as p:
        out = p.map(job, jobs)
    agg = {}
    for s, sv in out:
        agg.setdefault(s, []).append(sv)
    scores = sorted(((float(np.mean(v)), s, v) for s, v in agg.items()),
                    reverse=True)
    for m, s, v in scores:
        print(f"個体{s}: 平均{m:.2f}s ({', '.join(f'{x:.1f}' for x in v)})",
              flush=True)
    print(f"選抜: 個体{scores[0][1]}", flush=True)
    json.dump({"selected_seed": int(scores[0][1]),
               "model_survival": scores[0][0]},
              open("outputs/reward_K_selected.json", "w"))
    print("DONE", flush=True)
