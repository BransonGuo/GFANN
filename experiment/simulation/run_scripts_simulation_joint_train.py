import subprocess
from datetime import datetime
import time
import os
import numpy as np
import multiprocessing
import os
import functools
import sys
from ast import literal_eval

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import *
if WORKPATH not in sys.path:
    sys.path.append(WORKPATH)

experiment_file = os.environ.get("SIM_EXPERIMENT_FILE", "exp_simulation.py")
batch_size = int(os.environ.get("SIM_BATCH_SIZE", "64"))
num_epoch = int(os.environ.get("SIM_NUM_EPOCH", "300"))
n_trials = int(os.environ.get("SIM_N_TRIALS", "100"))
patience = int(os.environ.get("SIM_PATIENCE", "30"))
min_delta = float(os.environ.get("SIM_MIN_DELTA", "0"))
script_path = os.path.dirname(__file__)
script_folder = os.path.basename(script_path)


def parse_int_list_env(env_name, default_values):
    value = os.environ.get(env_name)
    if not value:
        return list(default_values)
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(parsed, int):
        parsed = [parsed]
    return [int(item) for item in parsed]


def parse_str_list_env(env_name, default_values):
    value = os.environ.get(env_name)
    if not value:
        return list(default_values)
    try:
        parsed = literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    return [str(item).strip().lower() for item in parsed if str(item).strip()]


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
    # 'pCA_NN_PCAOpt',
    # 'pCA_NN_ADD_PCAOpt'
    # #  SOFT================================
    'sPCA_NNOpt',
    # 'nN_SPCA_NNOpt',
    # "sPCA_NN_SPCA_ADDOpt",
]


def run_one_setting(seed, log_dir, p, factor_id, hcm_id, y_func_l, model_l_comma, setting):
    log_dir_new = f"Simulation/{log_dir}/{setting}/trial_{n_trials}/p_{p}/seed{seed}"
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
        str(num_epoch),
        "--batch_size",
        str(batch_size),
        "--patience",
        str(patience),
        "--min_delta",
        str(min_delta),
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
    if setting == "joint":
        cmd.extend(["--joint_train", "True"])
    subprocess.call(cmd)


def run_script(seed, log_dir):
    y_func_l = list(np.random.randint(0, 8, 10))
    model_l_comma = ','.join(model_l)
    p_l = parse_int_list_env("SIM_P_LIST", [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000])
    run_modes = parse_str_list_env("SIM_RUN_MODES", ["joint", "freeze"])
    invalid_modes = sorted(set(run_modes) - {"joint", "freeze"})
    if invalid_modes:
        raise ValueError(f"Invalid SIM_RUN_MODES values: {invalid_modes}. Use joint, freeze, or joint,freeze.")
    print(f"seed={seed}, y_func_l={y_func_l}, p_l={p_l}, run_modes={run_modes}")
    for p in p_l:
        for factor_id in [0, 1]:
            for hcm_id in [2]:
                for setting in run_modes:
                    run_one_setting(seed, log_dir, p, factor_id, hcm_id, y_func_l, model_l_comma, setting)

if __name__ == "__main__":
    multiprocess = True
    experiment_name = os.path.splitext(os.path.basename(experiment_file))[0]
    suffix = (
        f'Simluation_AllDim_joint_22_add_speed_b{batch_size}_'
        f'pat{patience}_{experiment_name}_{script_folder}'
    )
    start_time = time.time()
    log_dir = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + suffix
    text = f'START {suffix} \n {__file__}'

    if multiprocess:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        seeds = parse_int_list_env("SIM_SEEDS", range(100, 150, 5))
        pool_size = int(os.environ.get("SIM_POOL_SIZE", "1"))
        print(
            f"Running seeds={seeds}, p_l={parse_int_list_env('SIM_P_LIST', [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000])}, "
            f"run_modes={parse_str_list_env('SIM_RUN_MODES', ['joint', 'freeze'])}, "
            f"pool_size={pool_size}, log_dir={log_dir}"
        )
        p = multiprocessing.Pool(pool_size)
        result = p.map(functools.partial(run_script, log_dir=log_dir), seeds)
        p.close()
        p.join()
        
    print('time taken:', time.time() - start_time)
    time_taken = time.time() - start_time
    text = f'END {suffix} \n {__file__} \
        \nTime taken is {time_taken//60} min'

    
