import os
import sys

import joblib
from colorama import init, Fore
import torch
import random
import numpy as np
from torch import nn
import shutil
from scipy.sparse.linalg import eigsh as largest_eigsh
import functools
import argparse
import time
from datetime import datetime
import pandas as pd
import pathlib
import copy
from typing import Any, Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from methods.stat_methods_soft import *
import methods.stat_methods as stat_methods
from models.model_lib_soft_PCA import *
from config import *

if WORKPATH not in sys.path:
    sys.path.append(WORKPATH)
from logs import log
from utils.utils import *

init(autoreset=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n", help="number of samples", type=int, default=500)
parser.add_argument(
    "--m",
    help="number of samples to calculate the diversified " "projection matrix",
    type=int,
    default=256,
)
parser.add_argument("--p", help="data dimension", type=int, default=1000)
parser.add_argument("--r", help="factor dimension", type=int, default=5)
parser.add_argument(
    "--r_bar", help="diversified weight dimension", type=int, default=10
)
parser.add_argument("--width", help="width of NN", type=int, default=300)
parser.add_argument("--depth", help="depth of NN", type=int, default=3)
parser.add_argument("--add_width", help="width of add", type=int, default=10)
parser.add_argument("--add_depth", help="depth of add", type=int, default=2)
parser.add_argument("--nn_depth", help="nn_depth", type=int, default=-1)
parser.add_argument("--seed", help="random seed of numpy", type=int, default=150)
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-2)
parser.add_argument("--dropout_rate", help="dropout rate", type=float, default=0.6)
parser.add_argument("--exp_id", help="exp id", type=int, default=1)
parser.add_argument(
    "--record_dir", help="directory to save record", type=str, default=""
)
parser.add_argument("--log_dir", help="directory to save log", type=str, default="")
parser.add_argument("--suffix", help="suffix of the log file", type=str, default="")
parser.add_argument("--memo", help="memo describing the log file", type=str, default="")
parser.add_argument("--noise", help="noise level", type=float, default=1)
parser.add_argument("--b_f", help="factor bound", type=float, default=1)
parser.add_argument("--b_u", help="factor noise bound", type=float, default=1)
parser.add_argument("--num_epoch", help="num_epoch", type=int, default=200)
parser.add_argument("--factor_id", help="factor_id", type=int, default=200)
parser.add_argument("--hcm_id", help="hcm_id", type=int, default=200)
parser.add_argument("--n_trials", help="n_trials", type=int, default=50)
parser.add_argument(
    "--summary_file", help="summary_file", type=str, default="summary_file"
)
parser.add_argument(
    "--linear",
    help="linear factor model",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--reformat_res",
    help="reformat csv res",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--use_scheduler_step",
    help="whether to call scheduler.step() in NN optimization loops",
    default=True,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--use_proj_mean",
    help="take average of the projection matrix",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--record_proj",
    help="record projection matrix",
    default=False,
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--x_func_l", help="x_func_l", type=str, default="[0,1,0,1,0,1,0,1,0,1]"
)
parser.add_argument(
    "--y_func_l", help="y_func_l", type=str, default="[0, 1,2,3,4,5,6,7]"
)
parser.add_argument("--reg_lambda", help="reg_lambda", type=float, default=0)
parser.add_argument("--train_window", help="train_window", type=int, default=252)
parser.add_argument("--valid_window", help="valid_window", type=int, default=40)
parser.add_argument("--test_window", help="test_window", type=int, default=252)
parser.add_argument("--start_test_day", help="start_test_day", type=int, default=450)
parser.add_argument("--start_test_date", help="start_test_date", type=str, default="")
parser.add_argument("--data_file", help="data_file", type=str, default="")
parser.add_argument("--fred_idx", help="fred_idx", type=int, default=87)
parser.add_argument("--model_l", help="model_l", type=str, default="[]")
args = parser.parse_args()
######overwrite session#######
MAX_DEPTH = 4
args.loss_type = 'var'
args.model_l = args.model_l.split(',')
args.init_schedule = list(range(50)) 
args.init_schedule_ori = list(range(1, 50))
args.save_study = False
args.opt = True
# fit_by_epochs means if want to refit the model after optuna, and without validation data
args.retrain = False
args.fit_by_epochs = False
args.analyze = False
args.use_loss = True
args.batch_size = args.test_window
args.reformat_res = False
args.m = args.n
# hyper-parameters
args.rolling_train = True
args.delay = False
args.rank = False
args.normalize = False
args.fred_data = False

if args.use_proj_mean:
    args.record_proj = True
start_time = time.time()
if len(args.suffix) == 0:
    args.suffix = f'n{args.n}b{args.batch_size}noi{args.noise}'
suffix = datetime.fromtimestamp(start_time).strftime("%y%m%d-%H%M%S.%f") + str(args.seed) + str(args.lr) + args.suffix

# set random seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# args.seed = 166
seed_everything(seed=args.seed)


model_l = [

            ]

if len(args.model_l) > 0:
    model_l = args.model_l
train_window, valid_window, test_window = args.train_window, args.valid_window, args.test_window
start_test_day = args.start_test_day
assert start_test_day >= train_window + valid_window

df = pd.read_csv(fr'{args.data_file}')

df.columns = ['date'] + list(df.columns[1:])
df.set_index(['date'], inplace=True)
dff = df.dropna()
try:
    dff.index = pd.to_datetime(dff.index, format='%Y%m%d')
except:
    dff.index = pd.to_datetime(dff.index, format='%Y-%m-%d')
dff.sort_index(inplace=True)
if start_test_day > 0:
    if len(args.start_test_date) > 0:
        start_test_day = dff.index.get_loc(args.start_test_date)
    dff = dff[start_test_day - train_window - valid_window:]
args.p = dff.shape[1] - 1
x_all, y_all = dff.iloc[:,:-1].values, dff.iloc[:,-1:].values
#===================================================================
if args.rank:
    x_all = pd.DataFrame(x_all).rank(axis=1).values


def fill_nan_with_row_mean(array):
    # Iterate over each row in the array
    for i in range(array.shape[0]):
        # Calculate mean excluding NaN
        mean_val = np.nanmean(array[i])
        # Replace NaNs in the row with the computed mean
        array[i, np.isnan(array[i])] = mean_val
    return array

def merge_dic_to_df(best_valid_loss_dic, best_valid_score_dic, test_loss_dic, test_score_dic):
    best_valid_loss_dic_temp = {f'{k}_valid': v for k, v in best_valid_loss_dic.items()}
    best_valid_score_dic_temp = {f'{k}_valid_score': v for k, v in best_valid_score_dic.items()}
    test_loss_dic_temp = {f'{k}_test': v for k, v in test_loss_dic.items()}
    test_score_dic_temp = {f'{k}_test_score': v for k, v in test_score_dic.items()}
    summary_dict = dict(args.__dict__, **best_valid_loss_dic_temp, **best_valid_score_dic_temp, **test_loss_dic_temp, **test_score_dic_temp)
    df = pd.DataFrame([summary_dict])
    return df

def _aligned_warmup_y(warmup_pred: np.ndarray, warmup_y: np.ndarray) -> np.ndarray:
    warmup_len = len(np.asarray(warmup_pred).reshape(-1))
    return warmup_y[-warmup_len:]


def write_summary(
    res_df,
    res_by_score_df,
    test_pred_dic,
    test_pred_by_score_dic,
    pred_all_y,
    model_l,
    warmup_pred_dic=None,
    warmup_pred_by_score_dic=None,
    warmup_y=None,
    verbose=False,
    include_benchmark=True,
):
    output_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent) + f'/{args.summary_file}.csv'
    # output_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent) + f'\\{suffix}.csv'
    output_path_by_score = output_path[:-4] + '_by_score.csv'
    if not args.fred_data:
        for model in model_l:
            warmup_signal = None
            warmup_signal_by_score = None
            warmup_y_model = None
            if warmup_pred_dic is not None and model in warmup_pred_dic:
                warmup_signal = pd.DataFrame(warmup_pred_dic[model])
                warmup_y_model = _aligned_warmup_y(warmup_signal, warmup_y)
            if warmup_pred_by_score_dic is not None and model in warmup_pred_by_score_dic:
                warmup_signal_by_score = pd.DataFrame(warmup_pred_by_score_dic[model])
                if warmup_y_model is None:
                    warmup_y_model = _aligned_warmup_y(warmup_signal_by_score, warmup_y)

            res_temp = results_analytics_(
                pd.DataFrame(test_pred_dic[model]),
                pred_all_y,
                warmup_signal=warmup_signal,
                warmup_y=warmup_y_model,
            )
            res_temp.columns = [f'{model}_' + x for x in res_temp.columns]
            res_df = pd.concat([res_df, res_temp], axis=1)

            res_by_score_temp = results_analytics_(
                pd.DataFrame(test_pred_by_score_dic[model]),
                pred_all_y,
                warmup_signal=warmup_signal_by_score,
                warmup_y=warmup_y_model,
            )
            res_by_score_temp.columns = [f'{model}_' + x for x in res_by_score_temp.columns]
            res_by_score_df = pd.concat([res_by_score_df, res_by_score_temp], axis=1)
        if include_benchmark:
            res_dic = {}
            ret_series = calc_ret_series_from_pos(pred_all_y-pred_all_y+1, pred_all_y)
            res_dic['sharpe_ratio'] = calc_sharpe_ratio(ret_series)
            res_dic['pct_max_dd'] = calc_max_dd(ret_series)
            res_temp = pd.DataFrame(res_dic)
            res_temp.columns = ['buy_hold_' + x for x in res_temp.columns]
            res_df = pd.concat([res_df, res_temp], axis=1)
    if verbose:
        print(res_df.T)
    if not args.reformat_res:
        res_df.to_csv(output_path, index=False, mode='a', header=True)
        res_by_score_df.to_csv(output_path_by_score, index=False, mode='a', header=True)
    else:
        if os.path.exists(output_path):
            df_existing = pd.read_csv(output_path)
            res_df = pd.concat([df_existing, res_df])
        if os.path.exists(output_path_by_score):
            df_by_score_existing = pd.read_csv(output_path_by_score)
            res_by_score_df = pd.concat([df_by_score_existing, res_by_score_df])
        res_df.to_csv(output_path, index=False)
        res_by_score_df.to_csv(output_path_by_score, index=False)

def merge_res(res_l):
    metric_l = ['valid_loss', 'valid_score', 'test_loss', 'test_score']
    (
        res_df,
        res_by_score_df,
        test_pred_dic,
        test_pred_by_score_dic,
        warmup_pred_dic,
        warmup_pred_by_score_dic,
    ) = copy.deepcopy(res_l[0])
    metric_col_count = len(metric_l) * len(model_l)
    metric_cols = res_df.columns[-metric_col_count:]
    res_df[metric_cols] = res_df[metric_cols].astype("float64")
    res_by_score_df[metric_cols] = res_by_score_df[metric_cols].astype("float64")
    res_df[pd.isna(res_df)]=0
    res_by_score_df[pd.isna(res_by_score_df)]=0
    for (
        res_df_temp,
        res_by_score_df_temp,
        test_pred_dic_temp,
        test_pred_by_score_dic_temp,
        _warmup_pred_dic_temp,
        _warmup_pred_by_score_dic_temp,
    ) in res_l[1:]:
        res_df_temp[pd.isna(res_df_temp)]=0
        res_by_score_df_temp[pd.isna(res_by_score_df_temp)]=0
        res_df.iloc[:, -metric_col_count:] = res_df.iloc[:, -metric_col_count:] + res_df_temp.iloc[:, -metric_col_count:]
        res_by_score_df.iloc[:, -metric_col_count:] = res_by_score_df.iloc[:, -metric_col_count:] \
                                                      + res_by_score_df_temp.iloc[:,-metric_col_count:]
        for model in model_l:
            test_pred_dic[model] = np.vstack([test_pred_dic[model], test_pred_dic_temp[model]])
            test_pred_by_score_dic[model] = np.vstack([test_pred_by_score_dic[model], test_pred_by_score_dic_temp[model]])
    res_df.iloc[:, -metric_col_count:] /= len(res_l)
    res_by_score_df.iloc[:, -metric_col_count:] /= len(res_l)
    return (
        res_df,
        res_by_score_df,
        test_pred_dic,
        test_pred_by_score_dic,
        warmup_pred_dic,
        warmup_pred_by_score_dic,
    )


def _to_numpy_prediction(pred: Any) -> np.ndarray:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    return np.asarray(pred).reshape(-1)


def _predict_validation_segment(
    model: Any,
    x_valid_obs: np.ndarray,
    y_valid: np.ndarray,
    x_train_obs: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    try:
        return _to_numpy_prediction(
            model.predict(torch.tensor(x_valid_obs, dtype=torch.float32).to(device))
        )
    except Exception:
        try:
            return _to_numpy_prediction(model.predict(x_valid_obs))
        except Exception:
            if isinstance(model, DiffusionIndexARAdapter):
                return _to_numpy_prediction(
                    model.predict(
                        x_valid_obs,
                        y_true=np.asarray(y_valid).reshape(-1),
                        y_history=np.asarray(y_train).reshape(-1),
                        X_history=np.asarray(x_train_obs),
                    )
                )
            if isinstance(model, (ARPAdapter, EWMAAdapter)):
                return _to_numpy_prediction(
                    model.predict(
                        x_valid_obs,
                        y_true=np.asarray(y_valid).reshape(-1),
                        y_history=np.asarray(y_train).reshape(-1),
                    )
                )
            raise


def _predict_direct_segment(model: Any, x_obs: np.ndarray) -> np.ndarray:
    try:
        return _to_numpy_prediction(
            model.predict(torch.tensor(x_obs, dtype=torch.float32).to(device))
        )
    except Exception:
        return _to_numpy_prediction(model.predict(x_obs))


def _predict_train_valid_warmup(model: Any) -> np.ndarray:
    x_warmup = np.vstack([x_train_obs, x_valid_obs])
    try:
        return _predict_direct_segment(model, x_warmup).reshape(-1, 1)
    except Exception:
        # AR/DI-style models need lag history, so validation is the first clean
        # warm-up segment available without using pre-sample realized y.
        return _predict_validation_segment(
            model,
            x_valid_obs,
            y_valid.squeeze(),
            x_train_obs,
            y_train.squeeze(),
        ).reshape(-1, 1)


def record_model_performance(model_name, model_path=None):
    pred = models_dic[model_name].fit_and_predict(x_train_obs, y_train.squeeze(), x_valid_obs,
                                                        y_valid.squeeze(),
                                                        x_test_obs, n_jobs=1,
                                                        study_name=f'{models_dic[model_name].__class__.__name__}_{suffix}',
                                                        retrain=args.retrain,
                                                        fit_by_epochs=args.fit_by_epochs,
                                                        y_test=y_test.squeeze(),
    )
    pred = pred.reshape(-1,1)
    warmup_pred = _predict_train_valid_warmup(models_dic[model_name])
    pred_valid = None
    try:
        pred_valid = _predict_validation_segment(
            models_dic[model_name],
            x_valid_obs,
            y_valid.squeeze(),
            x_train_obs,
            y_train.squeeze(),
        )
        valid_loss = mse_loss(torch.from_numpy(pred_valid).reshape(-1, 1).to(device),
                              torch.from_numpy(y_valid.squeeze()).reshape(-1, 1).to(device)).item()
    except:
        valid_loss = models_dic[model_name].global_best_valid_loss
    test_loss = mse_loss(torch.from_numpy(pred).reshape(-1, 1), torch.from_numpy(y_test.squeeze()).reshape(-1, 1)).item()
    if args.fred_data:
        if torch.is_tensor(pred_valid):
           pred_valid = pred_valid.detach().cpu().numpy()
        tss_this_run = np.mean(np.square(y_valid))
        pred_recon = pred_valid.reshape(-1,1)
        y_valid_ = y_valid.reshape(-1,1)
        assert y_valid_.shape == pred_recon.shape
        rss_t = np.mean(np.square(y_valid_ - pred_recon))
        r_sqr = 1 - rss_t / tss_this_run
        valid_score = -r_sqr

        tss_this_run = np.mean(np.square(y_test))
        pred_recon = pred.reshape(-1,1)
        y_test_ = y_test.reshape(-1,1)
        assert y_test_.shape == pred_recon.shape
        rss_t = np.mean(np.square(y_test_ - pred_recon))
        r_sqr = 1 - rss_t / tss_this_run
        test_score = -r_sqr
    else:
        try:
            benchmark_std = models_dic[model_name].predict(torch.tensor(data[0][0], dtype=torch.float32).to(device)).std().item()
            valid_score = getattr(models_dic[model_name], "best_valid_score", None)
        except:
            benchmark_std = float(np.std(pred_valid)) if pred_valid is not None else 0.0
            valid_score = getattr(models_dic[model_name], "best_valid_score", None)
        if valid_score is None:
            valid_score = valid_loss if valid_loss is not None else 999.0
        ret_series = calc_ret_series(pred, y_test)
        test_score = -calc_sharpe_ratio(ret_series)#.values[0]
    if args.save_study:
        joblib.dump(models_dic[model_name].study,
                    f'{model_path}/{models_dic[model_name].__class__.__name__}_study_{suffix}.pkl')
        if torch.nn.modules.module.Module in models_dic[model_name].model.__class__.__bases__:
            models_dic[model_name].model.load_state_dict(models_dic[model_name].best_state_dict)
            # Export the model to ONNX
            onnx_path = f"{model_path}/{models_dic[model_name].__class__.__name__}_{suffix}.onnx"
            torch.onnx.export(models_dic[model_name].model,
                                  torch.from_numpy(x_test_obs[:1]).to(torch.float32).to('cuda'),
                                  onnx_path, verbose=False)
            from onnx2torch import convert
            import onnx
            onnx_model = onnx.load(onnx_path)
            torch_model = convert(onnx_model)
            torch_model.eval()
            pred_load = torch_model(torch.from_numpy(x_test_obs).to(torch.float32))
            test_loss_load = mse_loss(pred_load, torch.from_numpy(y_test.squeeze()).reshape(-1, 1))
            # assert round(test_loss.item(), 4) == round(test_loss_load.item(), 4)
    try:
        model_kwargs = copy.deepcopy(models_dic[model_name].best_model_kwargs)
        model_kwargs['best_epoch'] = models_dic[model_name].best_epoch
        model_kwargs['global_best_valid_loss'] = models_dic[model_name].global_best_valid_loss
        model_kwargs.update(models_dic[model_name].study.best_params)
        model_kwargs['benchmark_std'] = benchmark_std
    except:
        model = models_dic[model_name]
        model_kwargs = {
            attr: getattr(model, attr)
            for attr in ["alpha", "halflife", "global_best_valid_loss"]
            if hasattr(model, attr)
        }
        if hasattr(model, "halflife_grid"):
            model_kwargs["halflife_grid"] = getattr(model, "halflife_grid")
        if "benchmark_std" in locals():
            model_kwargs["benchmark_std"] = benchmark_std
    return pred, warmup_pred, valid_loss, valid_score, test_loss, test_score, model_kwargs


def joint_train(model_names, logger, parallel=False):
    colors = [Fore.RED, Fore.YELLOW, Fore.BLUE, Fore.GREEN, Fore.CYAN, Fore.LIGHTRED_EX, Fore.LIGHTYELLOW_EX,
              Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTRED_EX,
              Fore.LIGHTWHITE_EX, Fore.LIGHTBLACK_EX, Fore.GREEN, Fore.CYAN, ]
    model_color = {}
    for i, name in enumerate(model_names):
        model_color[name] = colors[i]
    valid_loss_dic, valid_score_dic = {}, {}
    test_loss_dic, test_score_dic = {}, {}
    best_param_dic = {}
    test_pred_dic = {}
    warmup_pred_dic = {}
    model_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent)
    for model_name in model_names:
        test_pred_dic[model_name], warmup_pred_dic[model_name], valid_loss_dic[model_name], valid_score_dic[model_name], test_loss_dic[model_name], \
        test_score_dic[model_name], best_param_dic[model_name] = record_model_performance(model_name, model_path=model_path)

    logger.info(f"valid_loss_dic {valid_loss_dic}, valid_score_dic {valid_score_dic}, test_loss_dic {test_loss_dic}, test_score_dic {test_score_dic}")
    res_df = merge_dic_to_df(valid_loss_dic, valid_score_dic, test_loss_dic, test_score_dic)
    return (
        res_df,
        copy.deepcopy(res_df),
        test_pred_dic,
        copy.deepcopy(test_pred_dic),
        warmup_pred_dic,
        copy.deepcopy(warmup_pred_dic),
        best_param_dic,
    )


def train_space(trial: Any) -> Dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
        "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [args.train_window]),
    }


def train_space_reg_var(trial: Any, pcaa: bool = False) -> Dict[str, Any]:
    trial_dict = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
        "optimizer_name": trial.suggest_categorical("optimizer", ["Adam"]),
        "batch_size": trial.suggest_categorical("batch_size", [args.train_window]),
        "lambda_orthogonality": trial.suggest_float(
            "lambda_orthogonality", 1e-6, 1, log=True
        ),
        "lambda_pca": trial.suggest_float("lambda_pca", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 1),
    }
    if pcaa:
        trial_dict["lambda_weight"] = trial.suggest_categorical(
            "lambda_weight", [0, 0.001, 0.003]
        )
    return trial_dict


def train_space_sparsity(trial: Any) -> Dict[str, Any]:
    return {
        "lambda_sparsity": trial.suggest_categorical(
            "lambda_sparsity",
            [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
        ),
    }


def model_space(trial: Any, depth_range: Tuple[int, int] = (3, 6)) -> Dict[str, Any]:
    r_bar = trial.suggest_int("r_bar", 20, 20)
    return {
        "r_bar": r_bar,
        "depth": trial.suggest_int("depth", depth_range[0], depth_range[1]),
        "width": trial.suggest_int("width", 300, 300),
        "check_depth": True,
    }


def model_space_add(
    trial: Any, depth_range: Tuple[int, int] = (3, 6), min_depth: int = 2
) -> Dict[str, Any]:
    r_bar = trial.suggest_int("r_bar", 20, 20)
    add_depth = trial.suggest_int("add_depth", 2, 2)
    depth = trial.suggest_int(
        "depth", add_depth + min_depth, max(add_depth + min_depth, depth_range[1])
    )
    return {
        "r_bar": r_bar,
        "depth": depth,
        "width": trial.suggest_int("width", 300, 300),
        "add_width": trial.suggest_int("add_width", 3, 20),
        "add_depth": add_depth,
        "check_depth": True,
    }


def model_space_bottleneck(
    trial: Any, depth_range: Tuple[int, int] = (3, 6)
) -> Dict[str, Any]:
    r_bar = trial.suggest_int("r_bar", 20, 20)
    return {
        "r_bar": r_bar,
        "depth": trial.suggest_int("depth", depth_range[0], depth_range[1]),
        "width": trial.suggest_int("width", 300, 300),
        "bottleneck_width": trial.suggest_int("bottleneck_width", 3, 20),
        "check_depth": True,
        "input_dropout": False,
    }


def build_models_dic(
    args: argparse.Namespace,
    model_params: Dict[str, Any],
    model_params_ori: Dict[str, Any],
    model_params_dp: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "ewma": EWMAAdapter(halflife_grid=[20, 60, 120]),
        "lasso": Lasso(),
        "pcr": PCR(),
        "pls": PLS(),
        "arp": ARPAdapter(p_grid=list(range(21))),
        "di": DiffusionIndexARAdapter(
            p_grid=list(range(21)), k_grid=list(range(1, 11)), factor_lags=1
        ),
        "vanillaNNOpt": VanillaNNOpt(
            trial_train=train_space,
            trial_model=functools.partial(model_space, depth_range=(3, MAX_DEPTH)),
            **model_params,
        ),
        "autoencoderOpt": AutoencoderOpt(
            trial_train=train_space,
            trial_model=functools.partial(
                model_space_bottleneck, depth_range=(3, MAX_DEPTH)
            ),
            **model_params,
        ),
        "fAR-NNOpt": FactorAugmentedNNOpt(
            trial_train=train_space,
            trial_model=functools.partial(model_space, depth_range=(3, MAX_DEPTH)),
            **model_params_dp,
        ),
        "pCA_NN_PCA_ADDOpt": stat_methods.PCA_NN_PCA_ADDOpt(
            trial_train=train_space,
            trial_model=functools.partial(
                model_space_add, depth_range=(3, MAX_DEPTH),
            ),
            loss_type=args.loss_type,
            **model_params_ori,
        ),
        "sPCA_NN_SPCA_ADDOpt": PCA_NN_PCA_ADDOpt(
        trial_train=train_space_reg_var,
        trial_model=functools.partial(
            model_space_add, depth_range=(3, MAX_DEPTH),
        ),
        init_with_eye=False,
        loss_type=args.loss_type,
        **model_params,
        ),
    }


def init_logger() -> Any:
    log_separately = True
    if args.log_dir == "" and log_separately:
        args.log_dir = suffix + f"ETF_seed{args.seed}_trial{args.n_trials}"

    os.makedirs(f"{WORKPATH}/logs/{args.log_dir}", exist_ok=True)
    script_path = os.path.dirname(__file__)
    script_folder = os.path.basename(script_path)

    try:
        shutil.copy(
            __file__,
            f"{WORKPATH}/logs/{args.log_dir}/{os.path.basename(__file__)}",
        )
        for file in os.listdir(script_path):
            if file.startswith('run_') and file.endswith('.py'):
                shutil.copy(
                    os.path.join(script_path, file),
                    f"{WORKPATH}/logs/{args.log_dir}/{file}"
                )
    except FileExistsError:
        pass

    experiment_logger = log(
        path=f"{WORKPATH}/logs/{args.log_dir}/",
        file="logs" + suffix.split("/")[-1],
    )
    experiment_logger.info("Train {}".format(args))
    return experiment_logger


def build_data_windows() -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    if not args.rolling_train:
        total_window = len(y_all)
        effective_test_window = total_window - train_window - valid_window
    else:
        effective_test_window = test_window
        total_window = train_window + valid_window + effective_test_window

    windows = [
        (x_all[i : i + total_window], y_all[i : i + total_window])
        for i in range(
            0, len(x_all) - total_window + effective_test_window, effective_test_window
        )
    ]

    if (
        len(windows) > 1
        and len(windows[-1][-1]) - train_window - valid_window
        < max(30, 0.2 * effective_test_window)
    ):
        windows[-2] = (
            np.vstack(
                [windows[-2][0], windows[-1][0][train_window + valid_window :]]
            ),
            np.vstack(
                [windows[-2][1], windows[-1][1][train_window + valid_window :]]
            ),
        )
        windows = windows[:-1]

    return windows, effective_test_window


def prepare_window_data(
    x_df: np.ndarray, y_df: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_obs_local, y_train_local = x_df[:train_window], y_df[:train_window]
    if args.delay:
        x_valid_obs_local = x_df[train_window : train_window + valid_window - 1]
        y_valid_local = y_df[train_window : train_window + valid_window - 1]
    else:
        x_valid_obs_local = x_df[train_window : train_window + valid_window]
        y_valid_local = y_df[train_window : train_window + valid_window]
    x_test_obs_local = x_df[train_window + valid_window :]
    y_test_local = y_df[train_window + valid_window :]

    if args.normalize:
        x_train_mean = x_train_obs_local.mean(axis=0)
        x_train_std = x_train_obs_local.std(axis=0)
        x_train_std[x_train_std == 0] = 100
        x_train_obs_local = (x_train_obs_local - x_train_mean) / x_train_std
        x_valid_obs_local = (x_valid_obs_local - x_train_mean) / x_train_std
        x_test_obs_local = (x_test_obs_local - x_train_mean) / x_train_std

        if np.isnan(x_train_obs_local).any():
            x_train_obs_local = fill_nan_with_row_mean(x_train_obs_local)
            x_valid_obs_local = fill_nan_with_row_mean(x_valid_obs_local)
            x_test_obs_local = fill_nan_with_row_mean(x_test_obs_local)

        if args.fred_data:
            y_train_mean, y_train_std = y_train_local.mean(), y_train_local.std()
            y_train_local = (y_train_local - y_train_mean) / y_train_std
            y_valid_local = (y_valid_local - y_train_mean) / y_train_std
            y_test_local = (y_test_local - y_train_mean) / y_train_std

    return (
        x_train_obs_local,
        y_train_local,
        x_valid_obs_local,
        y_valid_local,
        x_test_obs_local,
        y_test_local,
    )


def build_model_params(
    loss_fn: nn.Module,
    device_name: str,
    dp_matrix_large: np.ndarray,
    unlabelled_x: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    base_params = {
        "input_width": args.p,
        "n_trials": args.n_trials,
        "epoch": args.num_epoch,
        "random_seed": args.seed,
        "loss_fn": loss_fn,
        "device": device_name,
        "patience": 100,
        "analyze": args.analyze,
        "use_loss": args.use_loss,
    }
    model_params = dict(base_params, init_schedule=args.init_schedule)
    model_params_ori = dict(base_params, init_schedule=args.init_schedule_ori)
    model_params_dp = dict(
        base_params,
        init_schedule=args.init_schedule,
        dp_matrix=dp_matrix_large,
        unlabelled_x=unlabelled_x,
    )
    return model_params, model_params_ori, model_params_dp


def save_param_results(train_result: Tuple[Any, ...], logger: Any) -> Tuple[Any, ...]:
    param_df_l = []
    param_dic_l = train_result[-1]
    result_without_params = train_result[:-1]

    for model_name, param_dic in param_dic_l.items():
        cleaned_param_dic = {
            key: value
            for key, value in param_dic.items()
            if key not in {"dp_matrix", "rs_matrix", "unlabelled_x"}
        }
        cleaned_param_dic["model_name"] = model_name
        param_df_l.append(pd.DataFrame([cleaned_param_dic]))

    param_df = pd.concat(param_df_l, ignore_index=True)
    output_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent) + "/df_param.csv"

    # Different rolling windows can produce different Optuna parameter columns.
    # Rewrite the file with the union of columns each time so the CSV stays valid.
    if os.path.exists(output_path):
        try:
            existing_param_df = pd.read_csv(output_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            print(f"warning: rebuilding malformed parameter file at {output_path}")
            existing_param_df = pd.read_csv(
                output_path, engine="python", on_bad_lines="skip"
            )
        combined_columns = list(existing_param_df.columns)
        for column in param_df.columns:
            if column not in combined_columns:
                combined_columns.append(column)
        existing_param_df = existing_param_df.reindex(columns=combined_columns)
        param_df = param_df.reindex(columns=combined_columns)
        param_df_temp = pd.concat([existing_param_df, param_df], ignore_index=True)
    else:
        param_df_temp = param_df

    param_df_temp.to_csv(output_path, index=False)
    for model_name in result_without_params[3].keys():
        try:
            benchmark_std_0 = param_df_temp.loc[
                param_df_temp.model_name == model_name, "benchmark_std"
            ].iloc[0]
            benchmark_std = param_df.loc[
                param_df.model_name == model_name, "benchmark_std"
            ]
            scaler = benchmark_std_0 / benchmark_std
            result_without_params[3][model_name] = (
                result_without_params[3][model_name] * scaler.iloc[0]
            )
        except:
            pass

    return result_without_params


def save_prediction_outputs(
    test_pred_dic: Dict[str, np.ndarray],
    test_pred_by_score_dic: Dict[str, np.ndarray],
    pred_all_y: np.ndarray,
    effective_test_window: int,
    warmup_pred_dic: Dict[str, np.ndarray] = None,
    warmup_pred_by_score_dic: Dict[str, np.ndarray] = None,
    warmup_y: np.ndarray = None,
    warmup_date_index: pd.Index = None,
) -> None:
    df_pred = pd.DataFrame(
        np.hstack(list(test_pred_dic.values())), columns=test_pred_dic.keys()
    )
    df_pred.columns = [
        "_".join(
            [
                column,
                str(train_window),
                str(valid_window),
                str(effective_test_window),
                str(args.lr),
                str(args.seed),
            ]
        )
        for column in df_pred.columns
    ]

    df_pred_by_score = pd.DataFrame(
        np.hstack(list(test_pred_by_score_dic.values())),
        columns=test_pred_by_score_dic.keys(),
    )
    df_pred_by_score.columns = [
        "_".join(
            [
                column,
                str(train_window),
                str(valid_window),
                str(valid_window),
                str(args.lr),
                str(args.seed),
            ]
        )
        for column in df_pred_by_score.columns
    ]

    output_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent) + "/df_pred.csv"
    output_path_by_score = output_path[:-4] + "_by_score.csv"
    date_index = dff.index[-(len(y_all) - train_window - valid_window) :]

    if not os.path.exists(output_path):
        df_pred["y"] = pred_all_y
        df_pred["date"] = date_index
        df_pred.T.to_csv(output_path)
    else:
        df_pred.T.to_csv(output_path, mode="a", header=False)

    if not os.path.exists(output_path_by_score):
        df_pred_by_score["y"] = pred_all_y
        df_pred_by_score["date"] = date_index
        df_pred_by_score.T.to_csv(output_path_by_score)
    else:
        df_pred_by_score.T.to_csv(output_path_by_score, mode="a", header=False)

    if warmup_pred_dic is None or warmup_y is None or warmup_date_index is None:
        return

    def _write_warmup_predictions(pred_dic: Dict[str, np.ndarray], output_path: str) -> None:
        warmup_len = len(warmup_y)
        df_warmup = pd.DataFrame(index=np.arange(warmup_len))
        for model_name, pred in pred_dic.items():
            pred_arr = np.asarray(pred).reshape(-1)
            col = np.full(warmup_len, np.nan)
            col[-len(pred_arr):] = pred_arr
            df_warmup[
                "_".join(
                    [
                        model_name,
                        str(train_window),
                        str(valid_window),
                        "warmup",
                        str(args.lr),
                        str(args.seed),
                    ]
                )
            ] = col
        if not os.path.exists(output_path):
            df_warmup["y"] = np.asarray(warmup_y).reshape(-1)
            df_warmup["date"] = warmup_date_index
            df_warmup.T.to_csv(output_path)
        else:
            df_warmup.T.to_csv(output_path, mode="a", header=False)

    warmup_output_path = str(pathlib.Path(logger.handlers[1].baseFilename).parent) + "/df_pred_warmup.csv"
    warmup_output_path_by_score = warmup_output_path[:-4] + "_by_score.csv"
    _write_warmup_predictions(warmup_pred_dic, warmup_output_path)
    if warmup_pred_by_score_dic is not None:
        _write_warmup_predictions(warmup_pred_by_score_dic, warmup_output_path_by_score)


def main() -> None:
    global logger
    global data
    global x_train_obs
    global y_train
    global x_valid_obs
    global y_valid
    global x_test_obs
    global y_test
    global mse_loss
    global device
    global models_dic

    logger = init_logger()
    res_l = []
    data, effective_test_window = build_data_windows()

    for x_df, y_df in data:
        (
            x_train_obs,
            y_train,
            x_valid_obs,
            y_valid,
            x_test_obs,
            y_test,
        ) = prepare_window_data(x_df, y_df)

        mse_loss = nn.MSELoss()
        benchmark_squared_loss = (
            mse_loss(torch.tensor(y_test[:-1]), torch.tensor(y_test[1:]))
            * len(y_test[1:])
        )
        print("benchmark_squared_loss------", benchmark_squared_loss)

        unlabelled_x = x_train_obs[: args.m]
        cov_mat = np.cov(unlabelled_x.T)
        _, eigen_vectors = largest_eigsh(cov_mat, x_train_obs.shape[1], which="LM")
        dp_matrix_large = eigen_vectors / np.sqrt(x_train_obs.shape[1])
        dp_matrix = eigen_vectors[:, -args.r_bar :] / np.sqrt(x_train_obs.shape[1])
        print(f"Diversified projection matrix size {np.shape(dp_matrix)}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")

        model_params, model_params_ori, model_params_dp = build_model_params(
            loss_fn=mse_loss,
            device_name=device,
            dp_matrix_large=dp_matrix_large,
            unlabelled_x=unlabelled_x,
        )
        models_dic = build_models_dic(
            args=args,
            model_params=model_params,
            model_params_ori=model_params_ori,
            model_params_dp=model_params_dp,
        )

        if args.opt:
            train_result = joint_train(model_l, logger=logger)
            res_l.append(save_param_results(train_result, logger))

    (
        res_df,
        res_by_score_df,
        test_pred_dic,
        test_pred_by_score_dic,
        warmup_pred_dic,
        warmup_pred_by_score_dic,
    ) = merge_res(res_l)
    pred_all_y = y_all[-(len(y_all) - train_window - valid_window) :]
    warmup_y = y_all[: train_window + valid_window]
    warmup_date_index = dff.index[: len(warmup_y)]
    write_summary(
        res_df,
        res_by_score_df,
        test_pred_dic,
        test_pred_by_score_dic,
        pred_all_y,
        model_l=model_l,
        warmup_pred_dic=warmup_pred_dic,
        warmup_pred_by_score_dic=warmup_pred_by_score_dic,
        warmup_y=warmup_y,
        verbose=False,
    )
    save_prediction_outputs(
        test_pred_dic=test_pred_dic,
        test_pred_by_score_dic=test_pred_by_score_dic,
        pred_all_y=pred_all_y,
        effective_test_window=effective_test_window,
        warmup_pred_dic=warmup_pred_dic,
        warmup_pred_by_score_dic=warmup_pred_by_score_dic,
        warmup_y=warmup_y,
        warmup_date_index=warmup_date_index,
    )


if __name__ == "__main__":
    main()


