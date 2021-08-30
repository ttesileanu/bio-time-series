#! /usr/bin/env python
""" Run hyperparameter optimization using random datasets of snippets. """

import argparse
import pickle

import tqdm
import h5py

from typing import Tuple, Union, Sequence

from bioslds.regressors import BioWTARegressor, CrosscorrelationRegressor
from bioslds.batch import hyper_score_ar
from bioslds.hyperopt import random_maximize
from bioslds.dataset import RandomSnippetDataset
from bioslds.arma import make_random_arma
from bioslds.cluster_quality import unordered_accuracy_score
from bioslds.hdf import write_object_hierarchy


def slow_tqdm(*args, **kwargs):
    """ Return a tqdm progress bar with infrequent updates. """
    return tqdm.tqdm(mininterval=10, *args, **kwargs)


def make_bio_wta_with_stable_initial(*args, **kwargs) -> BioWTARegressor:
    """ Call the BioWTARegressor constructor, ensuring that the initial coefficients are
    chosen to correspond to stable AR processes.
    """
    weights = [
        make_random_arma(kwargs["n_features"], 0, rng=kwargs["rng"]).a
        for _ in range(kwargs["n_models"])
    ]
    return BioWTARegressor(*args, weights=weights, **kwargs)


def run_hyper_optimize(
    algo: str,
    dataset: RandomSnippetDataset,
    n_trials: int,
    n_features: Union[int, tuple],
    clusterer_seed: int,
    optimizer_seed: int,
    rate_range: tuple,
    rate_log: bool,
    exp_streak_range: tuple,
    exp_streak_log: bool,
    temperature_range: tuple,
    temperature_log: bool,
    timescale_range: tuple,
    timescale_log: bool,
    n_features_log: bool,
    feature_step_range: tuple,
    feature_step_log: bool,
    monitor: list,
    monitor_step: int,
    economy: bool,
) -> Tuple[float, dict, dict]:
    # handle log-scale options
    log_scale = []
    if rate_log:
        log_scale.append("rate")
    if exp_streak_log:
        log_scale.append("exp_streak")
    if temperature_log:
        log_scale.append("temperature")
    if timescale_log:
        log_scale.append("timescale")
    if n_features_log:
        log_scale.append("n_features")
    if feature_step_log:
        log_scale.append("feature_step")

    # handle int or tuple n_features
    if not hasattr(n_features, "__len__"):
        n_features_range = (n_features, n_features)
    else:
        n_features_range = n_features

    # choose some common options used for all algorithms when calling hyper_score_ar
    common_hyper_args = (dataset, unordered_accuracy_score)
    common_hyper_kws = dict(
        n_models=len(dataset.snippets),
        rng=clusterer_seed,
        progress=slow_tqdm,
        monitor=monitor,
        monitor_step=monitor_step,
    )

    # handle algorithm options
    if algo == "biowta":

        def fct(**kwargs):
            crt_res = hyper_score_ar(
                make_bio_wta_with_stable_initial,
                *common_hyper_args,
                n_features=kwargs["n_features"],
                rate=kwargs["rate"],
                trans_mat=1 - 1 / kwargs["exp_streak"],
                temperature=kwargs["temperature"],
                error_timescale=kwargs["timescale"],
                fit_kws={"step": kwargs["feature_step"]},
                **common_hyper_kws,
            )
            if economy:
                del crt_res[1].regressors
                if len(monitor) == 0:
                    del crt_res[1].history
            return crt_res

    elif algo == "xcorr":

        def fct(**kwargs):
            crt_res = hyper_score_ar(
                CrosscorrelationRegressor,
                *common_hyper_args,
                n_features=kwargs["n_features"],
                nsm_rate=kwargs["rate"],
                xcorr_rate=1 / kwargs["exp_streak"],
                fit_kws={"step": kwargs["feature_step"]},
                **common_hyper_kws,
            )
            if economy:
                del crt_res[1].regressors
                if len(monitor) == 0:
                    del crt_res[1].history
            return crt_res

    else:
        raise ValueError("Unknown algo.")

    res = random_maximize(
        fct,
        {
            "rate": rate_range,
            "exp_streak": exp_streak_range,
            "temperature": temperature_range,
            "timescale": timescale_range,
            "n_features": n_features_range,
            "feature_step": feature_step_range,
        },
        n_trials,
        log_scale=log_scale,
        rng=optimizer_seed,
        progress=slow_tqdm,
    )

    return res


class OpaqueList(list):
    """ List whose contents are not saved to HDF. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdf_skip_contents = True


def load_snippets(snip_type: str, snip_choice: str) -> OpaqueList:
    snip_file = f"{snip_type}_dataset.pkl"
    with open(snip_file, "rb") as f:
        all_snips = pickle.load(f)

    lst = OpaqueList()
    for item in snip_choice:
        lst.append(all_snips[item])

    return lst


def save_results(outfile: str, res: dict, force: bool):
    mode = "w" if force else "x"
    with h5py.File(outfile, mode) as f:
        write_object_hierarchy(f, res)


class StringedDatasetsIterator(object):
    def __init__(self, datasets: Sequence):
        self.idx = 0
        self.datasets = datasets
        self._starts = [len(self.datasets[0])]
        for dataset in self.datasets[1:]:
            self._starts.append(self._starts[-1] + len(dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self._starts[-1]:
            self.idx = 0
            raise StopIteration
    
        # find which dataset we're in
        for m, start in enumerate(self._starts):
            if start > self.idx:
                start = self._starts[m - 1] if m > 0 else 0
                break
        
        # find index in dataset
        i = self.idx - start

        self.idx += 1
        return self.datasets[m][i]
            


class StringedDatasets(object):
    def __init__(self, datasets: Sequence):
        self.datasets = datasets
        self._len = sum(len(_) for _ in self.datasets)
        
        # this is only to establish n_models in run_hyper_optimize
        self.snippets = ([], [])

    def __iter__(self) -> StringedDatasetsIterator:
        return StringedDatasetsIterator(self.datasets)

    def __len__(self) -> int:
        return self._len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for randomly generated datasets "
        "based on snippets."
    )

    parser.add_argument("outfile", help="where to save the results (HDF5)")
    parser.add_argument("snip_type", help="snippet dataset (vowel or pitch)")
    parser.add_argument(
        "snip_choice", help="snippets to use (aeiou for vowel, cdefgab for "
        "pitch; all_pairs to use all pairs -- note that then the actual number of"
        "signals becomes n_signals * n_pairs)"
    )
    parser.add_argument("algorithm", help="algorithm to use (biowta or xcorr)")

    parser.add_argument(
        "-n",
        "--n-trials",
        type=int,
        default=10,
        help="number of trials to use for optimization",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200000,
        help="number of time steps in each signal",
    )
    parser.add_argument(
        "--n-signals", type=int, default=10, help="number of signals to test on"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=4,
        help="number of features used by clusterer",
    )
    parser.add_argument(
        "-d", "--dataset", type=int, help="random seed for dataset generation"
    )
    parser.add_argument(
        "-c", "--clusterer", type=int, help="random seed for clusterer initialization"
    )
    parser.add_argument(
        "-o", "--optimizer", type=int, help="random seed for hyperparameter optimizer"
    )
    parser.add_argument(
        "--average-dwell",
        type=float,
        default=100.0,
        help="average dwell time in each latent state",
    )
    parser.add_argument(
        "--min-dwell",
        type=float,
        default=50,
        help="minimum dwell time in each latent state",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="ensure that each signal has unit variance",
    )
    parser.add_argument(
        "--rate-range",
        type=float,
        nargs=2,
        default=(0.05, 0.15),
        help="range for learning rate parameter",
    )
    parser.add_argument(
        "--rate-log",
        action="store_true",
        default=False,
        help="sample learning rate in log space",
    )
    parser.add_argument(
        "--exp-streak-range",
        type=float,
        nargs=2,
        default=(1.5, 50.0),
        help="range for expected streak length",
    )
    parser.add_argument(
        "--exp-streak-log",
        action="store_true",
        default=False,
        help="sample expected streak length in log space",
    )
    parser.add_argument(
        "--temperature-range",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        help="range for BioWTA temperature",
    )
    parser.add_argument(
        "--temperature-log",
        action="store_true",
        default=False,
        help="sample BioWTA temperature in log space",
    )
    parser.add_argument(
        "--timescale-range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        help="range for averaging timescale",
    )
    parser.add_argument(
        "--timescale-log",
        action="store_true",
        default=False,
        help="sample averaging timescale in log space",
    )
    parser.add_argument(
        "--n-features-range",
        type=int,
        nargs=2,
        default=None,
        help="range of number of features",
    )
    parser.add_argument(
        "--n-features-log",
        action="store_true",
        default=False,
        help="sample number of features in log space (rounded to int)",
    )
    parser.add_argument(
        "--feature-step-range",
        type=int,
        nargs=2,
        default=(1, 1),
        help="range of steps between features in lag vector",
    )
    parser.add_argument(
        "--feature-step-log",
        action="store_true",
        default=False,
        help="sample feature step in log space (rounded to int)",
    )
    parser.add_argument(
        "--store-signal-set",
        action="store_true",
        default=False,
        help="store the actual signals used",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="output progress info",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
        help="don't print anything",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="force overwrite existing file",
    )
    parser.add_argument(
        "--monitor", nargs="*", default=[], help="quantities to monitor"
    )
    parser.add_argument(
        "--monitor-step",
        type=int,
        default=1,
        help="how often to store monitored quantities",
    )
    parser.add_argument(
        "--economy",
        action="store_true",
        default=False,
        help="do not store regressor objects",
    )

    main_args = parser.parse_args()

    # perform some checks
    if main_args.algorithm not in ["biowta", "xcorr"]:
        exit("Unknown algorithm.")

    if main_args.verbose:
        if main_args.dataset is None:
            print("WARNING: no --dataset option, dataset seed is set to zero.")
            main_args.dataset = 0
        if main_args.clusterer is None:
            print("WARNING: no --clusterer option, clusterer seed is set to zero.")
            main_args.clusterer = 0
        if main_args.optimizer is None:
            print("WARNING: no --optimizer option, optimizer seed is set to zero.")
            main_args.optimizer = 0

    # generate dataset
    if main_args.snip_choice == "all_pairs":
        snip_names = {"vowel": "aeiou", "pitch": "cdefgab"}[main_args.snip_type]
    else:
        snip_names = main_args.snip_choice
        
    snippets = load_snippets(main_args.snip_type, snip_names)

    if main_args.snip_choice != "all_pairs":
        hyper_dataset = RandomSnippetDataset(
            main_args.n_signals,
            main_args.n_samples,
            snippets,
            dwell_times=main_args.average_dwell,
            min_dwell=main_args.min_dwell,
            normalize=main_args.normalize,
            rng=main_args.dataset,
        )
        if main_args.store_signal_set:
            hyper_dataset.hdf_skip_contents = False
    else:
        hyper_dataset = []
        for i in range(len(snip_names)):
            for j in range(i + 1, len(snip_names)):
                hyper_dataset.append(RandomSnippetDataset(
                    main_args.n_signals,
                    main_args.n_samples,
                    [snippets[i], snippets[j]],
                    dwell_times=main_args.average_dwell,
                    min_dwell=main_args.min_dwell,
                    normalize=main_args.normalize,
                    rng=main_args.dataset,
                ))
                if main_args.store_signal_set:
                    hyper_dataset[-1].hdf_skip_contents = False
        hyper_dataset = StringedDatasets(hyper_dataset)

    # run the hyper optimization
    n_feat = main_args.n_features
    if main_args.n_features_range is not None:
        n_feat = main_args.n_features_range
    hyper_res = run_hyper_optimize(
        main_args.algorithm,
        hyper_dataset,
        n_trials=main_args.n_trials,
        n_features=n_feat,
        clusterer_seed=main_args.clusterer,
        optimizer_seed=main_args.optimizer,
        rate_range=main_args.rate_range,
        rate_log=main_args.rate_log,
        exp_streak_range=main_args.exp_streak_range,
        exp_streak_log=main_args.exp_streak_log,
        temperature_range=main_args.temperature_range,
        temperature_log=main_args.temperature_log,
        timescale_range=main_args.timescale_range,
        timescale_log=main_args.timescale_log,
        n_features_log=main_args.n_features_log,
        feature_step_range=main_args.feature_step_range,
        feature_step_log=main_args.feature_step_log,
        monitor=main_args.monitor,
        monitor_step=main_args.monitor_step,
        economy=main_args.economy,
    )
    if main_args.verbose:
        print(f"Maximum median score: {hyper_res[0]:.2f}.")

    # store results, making sure to keep track of the parameters used
    to_save = hyper_res[2]
    to_save["main_args"] = main_args
    to_save["dataset"] = hyper_dataset
    to_save["best_scalar"] = hyper_res[0]
    to_save["best_params"] = hyper_res[1]
    save_results(main_args.outfile, to_save, force=main_args.force)
