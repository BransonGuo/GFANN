import subprocess
from datetime import datetime
import time
import os
import numpy as np
import multiprocessing
import os
import functools
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import *
if WORKPATH not in sys.path:
    sys.path.append(WORKPATH)

experiment_file = "exp_simulation.py"
script_path = os.path.dirname(__file__)
script_folder = os.path.basename(script_path)

model_l = [
    # benchmark================================
    # 'oracleNNOpt',
    # 'fAR-NNOpt',
    # 'vanillaNNOpt',
    # 'autoencoderOpt',
    # 'lasso',
    # 'pcr',
    # 'pls',
    # 'di',
    # 'arp',
    # # hard================================
    'pCA_NN_PCAOpt',
    'pCA_NN_ADD_PCAOpt',
    'pCA_NN_PCA_ADDOpt',
    # #  SOFT================================
    "sPCA_NN_SPCA_ADDOpt",
    'sPCA_NN_SPCAOpt',
    'sPCA_NN_ADD_SPCAOpt',
]

def run_script(seed, log_dir):
    y_func_l = list(np.random.randint(0, 8, 10))
    n_trials = 100
    model_l_comma = ','.join(model_l)
    for p in [500, 1000, 1500, 2000, 2500, 3000]:
        for factor_id in [0, 1]:
            for hcm_id in [2]:
                log_dir_new = f"Simulation/{log_dir}trial_{n_trials}/p_{p}/seed{seed}"
                cmd = [
                    sys.executable,
                    f"{script_path}/{experiment_file}",
                    "--suffix",
                    "experiments",
                    "--log_dir",
                    log_dir_new,
                    "--p",
                    str(p),
                    "--num_epoch",
                    "300",
                    "--n_trials",
                    str(n_trials),
                    "--y_func_l",
                    str(y_func_l),
                    "--model_l",
                    model_l_comma,
                    "--factor_id",
                    str(factor_id),
                    "--hcm_id",
                    str(hcm_id),
                    "--seed",
                    str(seed),
                ]
                subprocess.call(cmd)

if __name__ == "__main__":
    multiprocess = True
    suffix = f'Simluation_AllDim_ablation_{script_folder}'
    start_time = time.time()
    log_dir = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + suffix
    text = f'START {suffix} \n {__file__}'

    if multiprocess:
        seeds = range(100, 150, 5)
        # if use GPU, just set 1
        p = multiprocessing.Pool(1)
        result = p.map(functools.partial(run_script, log_dir=log_dir), seeds)
        
    print('time taken:', time.time() - start_time)
    time_taken = time.time() - start_time
    text = f'END {suffix} \n {__file__} \
        \nTime taken is {time_taken//60} min'

    
