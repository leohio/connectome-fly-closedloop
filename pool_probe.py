import sys
from multiprocessing import Pool, set_start_method

def trivial(i):
    return i * 2

def light(i):
    import reward_K as RK
    return RK.N_EP

def heavy(i):
    import numpy as np
    import reward_K as RK
    return RK.rollout_sensed(np.concatenate([RK.INNATE, np.zeros(RK.N_PAR)]),
                             0, T=0.3, noise_seed=i)[0]

if __name__ == "__main__":
    which = sys.argv[1]
    fn = {"trivial": trivial, "light": light, "heavy": heavy}[which]
    with Pool(2) as p:
        print(which, "->", p.map(fn, [0, 1]), flush=True)
