import subprocess
from datetime import datetime
import time
import os
import multiprocessing
import functools
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import *
if WORKPATH not in sys.path:
    sys.path.append(WORKPATH)
experiment_file = 'exp_FRED.py'
script_path = os.path.dirname(__file__)
script_folder = os.path.basename(script_path)

model_l = [
    # benchmark================================
    'fan_fast',
    'vanillaNNOpt',
    'lasso',
    'pcr',
    'pls',
    'di',
    'arp',
    # # #  SOFT================================
    'gFANNOpt',
]

# follow Fan's code
def run_script(fred_idx, log_dir):
    n_trials = 150
    model_l_comma = ','.join(model_l)
    log_dir_new = f"FRED/{log_dir}/fred_idx{fred_idx}_trial{n_trials}"
    cmd = [
        sys.executable,
        f"{script_path}/{experiment_file}",
        "--suffix",
        "experiments",
        "--log_dir",
        log_dir_new,
        "--num_epoch",
        "300",
        "--n_trials",
        str(n_trials),
        "--fred_idx",
        str(fred_idx),
        "--model_l",
        model_l_comma,
        "--seed",
        "2000",
        "--use_scheduler_step",
    ]
    subprocess.call(cmd)
        
if __name__ == "__main__":
    # subprocess.call(['python', "./far_exp.py --record_dir 'logs' --suffix 'test' --memo 'let us check this'"])
    multiprocess = True
    suffix = f'FRED_ALL_{script_folder}'
    start_time = time.time()
    log_dir = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + suffix
    text = f'START {suffix} \n {__file__}'

    mylist = list(range(127))#
    p = multiprocessing.Pool(1)
    result = p.map(functools.partial(run_script, log_dir=log_dir), mylist)
    print('time taken:', time.time() - start_time)
    time_taken = time.time() - start_time
    text = f'END {suffix} \n {__file__} \
        \nTime taken is {time_taken//60} min'

    
