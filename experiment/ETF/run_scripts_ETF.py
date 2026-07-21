import subprocess
import shlex
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

DATA_DIR = os.path.join(BASE_DIR, "data", "ETF_price", "SP Global")
file_l = [
    'cs_SimpleRet1-5-10-20-126_yLags1-20_SP500 as Y.csv.gz',
          ]
experiment_file = 'exp_ETF.py'
script_path = os.path.dirname(__file__)
script_folder = os.path.basename(script_path)
# if 2010, then start day = 2404

model_l = [
    # benchmark================================
    'fAR-NNOpt',
    'vanillaNNOpt',
    'autoencoderOpt',
    'ewma',
    'lasso',
    'pcr',
    'pls',
    'di',
    'arp',
    # # hard================================
    'pCA_NN_PCA_ADDOpt',
    # #  SOFT================================
    "sPCA_NN_SPCA_ADDOpt",
]

def run_script(log_dir, model):
    for seed in range(100, 200, 5):
        for n_trials in [20]:
            start_test_date = '2017-08-01'
            memo = ''
            model_l_comma = ','.join([model])
            for data_file in file_l:
                data_path = os.path.join(DATA_DIR, data_file)
                log_dir_new = f"ETF/{log_dir}_{start_test_date}{memo}trial{n_trials}/{data_file}/model_{model}"
                for train_window, valid_window, test_window in [(504, 60, 252)]:
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
                        "--model_l",
                        model_l_comma,
                        "--train_window",
                        str(train_window),
                        "--valid_window",
                        str(valid_window),
                        "--test_window",
                        str(test_window),
                        "--no-reformat_res",
                        "--seed",
                        str(seed),
                        "--data_file",
                        data_path,
                        "--start_test_day",
                        "630",
                        "--start_test_date",
                        start_test_date,
                    ]
                    subprocess.call(cmd)

if __name__ == "__main__":
    multiprocess = True
    suffix = f'SPGlobalBatchTrial20Seeds---{script_folder}'
    start_time = time.time()
    log_dir = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + suffix
    text = f'START {suffix} \n {__file__}'

    if multiprocess:
        p = multiprocessing.Pool(1)
        result = p.map(functools.partial(run_script, log_dir), model_l)
    print('time taken:', time.time() - start_time)

    
