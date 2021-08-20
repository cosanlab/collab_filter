"""
Usage: collab_filter/fit_all_models.py DATASET [options]

Python script that runs full estimation on all datasets using
parallelization. Depends on code/lib.py. This script should be called from the *project root*. All optional arguments will override the default estimation settings. Settings with the default of 'all' will vary depending on the exact dataset chosen, except for train-size which will default to 10-90% of the dataset size. Pass in list arguments using comma-separated values (no [] or '' required)

Examples:
    python code/fit_all_models.py moth --models=KNN,Mean
    python code/fit_all_models.py iaps --dry-run
    python code/fit_all_models.py decisions --factors=4,5,6 --models=NNMF_S,NNMF_M

Arguments:
    DATASET             Dataset to analyze. Must be one of [iaps, moth, decisions]

Options:
    -h, --help                  Print this help
    --dry-run                   Setup estimation but don't run it [default: False]
    --overwrite                 Overwrite output files [default: False]
    --clear-log                 Overwrite logfiles [default: False]
    --random-seed=SEED          Random seed (integer) for reproducibility [default: 516]
    --dilations=DILATIONS       Custom dilation values [default: all]
    --factors=FACTORS           Custom number of factors [default: all]
    --n-iter=N_ITER             Custom number of estimation iterations [default: 10]
    --train-sizes=TRAIN_SIZE    Custom training sizes [default: all]
    --dimensions=DIMENSIONS     Which dataset dimension to analyze [default: all]
    --models=MODELS             Which algorithms to use [default: all]
"""

# %%
# IMPORTS
import os
import sys
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from docopt import docopt
from neighbors import (
    KNN,
    Mean,
    NNMF_mult,
    NNMF_sgd,
    create_user_item_matrix,
    estimate_performance,
)
import ray
import logging
from filelock import FileLock

from lib import ProgressBar  # type: ignore

# %%

# FUNCTIONS
def save_results(user_results, dataset_name, save_dir):
    """Append the user results output of a call to `estimate_performance` to a two csv files"""
    user_results_path = save_dir / f"{dataset_name}_user_model_comparison.csv"
    user_results_lock = user_results_path.parent / f"{user_results_path.stem}.lock"
    if user_results is not None:
        with FileLock(user_results_lock):
            with open(user_results_path, "a") as f:
                user_results.to_csv(f, header=f.tell() == 0, index=False)


def prepare_data(data, dataset_name, dimension=None, average_trials=False):
    """
    Take a dataframe filtering by a particular value and return a users x items dataframe to serve as input to a model.

    Args:
        data (pd.DataFrame): dataframe of original data
        dataset_name (str): name of dataset

    Raises:
        ValueError: if dataset is not 'iaps', 'moth', 'decisions'

    Returns:
        [pd.DataFrame]: users x items dataframe
    """

    if dataset_name == "moth":
        ratings = create_user_item_matrix(
            data.query("movie == @dimension").reset_index(drop=True),
            ["workerId", "timeStamp", "ratingScore"],
        )
    elif dataset_name == "iaps":
        ratings = create_user_item_matrix(
            data.query("Appraisal_Dimension == @dimension").reset_index(drop=True),
            ["Subject", "Item", "Rating"],
        )
    elif dataset_name == "decisions":
        # For the vanBaar data we actually just want to grab a different column in the df rather than filter rows by a dimension
        if average_trials:
            data = data.groupby(["Subject", "Trial"], as_index=False)[dimension].mean()
        ratings = create_user_item_matrix(data, ["Subject", "Trial", dimension])
    else:
        raise ValueError("dataset_name should be moth, iaps, or decisions")

    return ratings


def setup_logger(filename):
    """Setup a logger to be used by a ray remote worker that logs to a file and optionally to std out"""

    logging.basicConfig(
        format="%(asctime)s | %(process)d | %(levelname)s | %(message)s",
        datefmt="%A-%b-%d at %I:%M:%S%p",
        filename=f"{filename}.log",
        level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.captureWarnings(True)


def setup_paths(args):
    base_dir = Path(os.getcwd())
    data_dir = base_dir / "data"
    analysis_dir = base_dir / "analysis"
    result_file = analysis_dir / f"{args['DATASET']}_overall_model_comparison.csv"
    user_result_file = analysis_dir / f"{args['DATASET']}_user_model_comparison.csv"

    # Handling existing output file
    if result_file.exists():
        if args["--overwrite"]:
            logging.info("Existing output file found. Removing...")
            result_file.unlink()
            user_result_file.unlink()
        else:
            logging.error(
                "Output file already exists. Use the --overwrite flag to forcibly overwrite it!"
            )
            sys.exit(1)
    return data_dir, analysis_dir, result_file, user_result_file


def setup_estimation(data_dir, args):

    parse_list_arg = lambda l: l.split(",") if "," in l else [l]
    # Parse args
    dataset = args["DATASET"]
    dimensions = (
        parse_list_arg(args["--dimensions"])
        if args["--dimensions"] != "all"
        else args["--dimensions"]
    )
    dilation = (
        parse_list_arg(args["--dilations"])
        if args["--dilations"] != "all"
        else args["--dilations"]
    )
    factors = (
        parse_list_arg(args["--factors"])
        if args["--factors"] != "all"
        else args["--factors"]
    )
    n_iter = int(args["--n-iter"])
    models = (
        ["Mean", "KNN", "NNMF_M", "NNMF_S"]
        if args["--models"] == "all"
        else parse_list_arg(args["--models"])
    )
    n_mask_items = (
        np.arange(0.1, 1.0, 0.1)
        if args["--train-sizes"] == "all"
        else parse_list_arg(args["--train-sizes"])
    )

    # Load up data
    if dataset not in ["iaps", "moth", "decisions"]:
        logging.error("dataset must be one of [iaps, moth, decisions]")
        sys.exit(1)
    data_file = data_dir / f"{dataset}.csv"
    if not data_file.exists():
        logging.error(
            "Datafile not found! Are you sure you ran this script from the *root directory* of the project (not the code sub-dir!)?"
        )
        sys.exit(1)
    else:
        data = pd.read_csv(data_file)

    # random seed for reproducibility
    SEED = int(args["--random-seed"])
    np.random.seed(SEED)

    # Dilation factor and dimensions
    # For each dataset the max number of factors is min(num_user, num_item)
    if dataset == "iaps":
        dilation = [None] if dilation == "all" else dilation
        if dimensions == "all":
            basic_emotions = ["anger", "sadness", "joy", "surprise", "fear", "disgust"]
            dimensions = list(
                filter(lambda x: x in basic_emotions, data.Appraisal_Dimension.unique())
            )
        if factors == "all":
            factors = [None, 25, 10, 5, 3, 2]

    elif dataset == "moth":
        dilation = [None, 5, 20, 60] if dilation == "all" else dilation
        dimensions = data.movie.unique() if dimensions == "all" else dimensions
        factors = [None, 20, 10, 5, 3, 2] if factors == "all" else factors

    elif dataset == "decisions":
        dilation = [None] if dilation == "all" else dilation
        dimensions = ["Prop_Returned"] if dimensions == "all" else dimensions
        factors = [None, 38, 19, 15, 9, 7, 4, 2] if factors == "all" else factors

    # Avoid fitting models that don't need certain args such as non NNMF models with factors
    def _filter_models(model_args):
        if model_args[4] == "NNMF_S" or model_args[4] == "NNMF_M":
            return True
        else:
            if model_args[3] is None:
                return True
            else:
                return False

    jobs = list(
        filter(
            _filter_models,
            list(product(dimensions, n_mask_items, dilation, factors, models)),
        )
    )

    seeds = np.random.randint(0, 9999, size=len(jobs))
    return (
        data,
        dataset,
        n_iter,
        n_mask_items,
        dimensions,
        factors,
        dilation,
        models,
        jobs,
        seeds,
    )


@ray.remote
def fit_all_algos(
    data, dataset_name, n_iter, model_map, fpath, progress_bar, seed, *args
):
    """
    Fits multiple algorithms to the same dataset given the same input parameters. Designed to be called in a (parallel) loop when each loop iteration provides a new seed and a tuple of (dataset dimension, n_mask_items, dilation)

    Args:
        data (pd.DataFrame): long-form data
        dataset_name (str): dataset name
        n_iter (int): number of estimation iterations
        model_map (dict): a dict of dicts containing each model to run
        fpath (Path): output directory
        seed (int): random seed for reproducibility
        *args (tup): (dimension, n_mask_items, dilation, factor, model_args)
    """

    # Create a new logger so this remote job can write to file
    setup_logger(dataset_name)

    # Parse args
    # Model hyper parameters
    model_map = {
        "Mean": {"model": Mean},
        "KNN": {"model": KNN},
        "NNMF_M": {"model": NNMF_mult},
        "NNMF_S": {"model": NNMF_sgd},
    }
    dimension, n_mask_items, dilation, factor, model_name = args
    fit_kwargs = model_map[model_name]
    algorithm = fit_kwargs.pop("model")
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if k != "model"}
    fit_kwargs["dilate_by_nsamples"] = dilation
    if model_name in ["NNMF_S", "NNMF_M"]:
        fit_kwargs["n_factors"] = factor
    else:
        logging.info(f"n_factors={factor} ignored for {model_name}")

    # Create users x items matrix
    ratings = prepare_data(data, dataset_name, dimension)

    try:
        _, user_results = estimate_performance(
            algorithm,
            ratings,
            n_iter,
            n_mask_items=n_mask_items,
            return_agg=False,
            return_full_performance=True,
            random_state=seed,
            fit_kwargs=fit_kwargs,
            verbose=False,
        )

        user_results["dilation"] = 0 if dilation is None else dilation
        user_results["train_size"] = 1 - n_mask_items
        user_results["dimension"] = dimension
        user_results["algorithm"] = model_name
        user_results["n_factors"] = "all" if factor is None else factor
        user_results["max_factors"] = min(ratings.shape)

    except Exception as e:
        logging.error(
            f"Fit failed on {model_name}, dimension: {dimension} n_mask_items: {n_mask_items}, dilation: {dilation} n_factors: {factor}"
        )
        logging.exception(e)
        user_results = None

    # Save
    save_results(user_results, dataset_name, fpath)
    logging.info(
        f"[Finished] dimension: {dimension} algorithm: {model_name} n_mask_items: {n_mask_items} dilation: {dilation} n_factors: {factor}"
    )
    progress_bar.update.remote(1)
    return


# %%
# SCRIPT
if __name__ == "__main__":

    # Parse cli args
    args = docopt(__doc__)

    # Setup root logger config
    if Path(f"{args['DATASET']}.log").exists() and args["--clear-log"]:
        Path(f"{args['DATASET']}.log").unlink()
    setup_logger(args["DATASET"])

    # Setup paths
    data_dir, analysis_dir, result_file, user_result_file = setup_paths(args)

    # Setup esimation
    (
        data,
        dataset,
        n_iter,
        n_mask_items,
        dimensions,
        factors,
        dilation,
        models,
        jobs,
        seeds,
    ) = setup_estimation(data_dir, args)

    # Start
    logging.info(
        f"[START (dryrun={args['--dry-run']})] dataset={args['DATASET']} overwrite={args['--overwrite']} clear-log={args['--clear-log']} random-seed={args['--random-seed']}"
    )
    logging.info(f"NUM ITERATIONS: {n_iter}")
    logging.info(f"NUM DIMENSIONS: {len(dimensions)} {dimensions}")
    logging.info(f"NUM MASK %: {len(n_mask_items)} {n_mask_items}")
    logging.info(f"NUM FACTORS: {len(factors)} {factors}")
    logging.info(f"NUM ALGOS: {len(models)} {models}")
    logging.info(f"NUM DILATION: {len(dilation)} {dilation}")
    logging.info(f"Unique parameter combinations (n_jobs): {len(jobs)}")

    if not args["--dry-run"]:
        # Initialize ray connection for parallel estimation
        try:
            ray.init(address="auto")
        except ConnectionError as e:
            print(
                "Whoops looks like you haven't started a ray instance which is required for parallel execution. Run the following from the command line before executing this script:\n"
            )
            print("ray start --head -v\n")
            sys.exit()

        # Run jobs
        pb = ProgressBar(len(jobs))
        actor = pb.actor
        futures = [
            fit_all_algos.remote(
                data,
                dataset,
                n_iter,
                models,
                analysis_dir,
                actor,
                seed,
                *job_args,
            )
            for seed, job_args in zip(seeds, jobs)
        ]
        pb.print_until_done()
        while len(futures):
            done, futures = ray.wait(futures)
            try:
                ray.get(done[0])
            except Exception as e:
                logging.exception(e)

        # Clean up lock files
        logging.info("CLEANING UP")
        _ = [e.unlink() for e in analysis_dir.glob("*.lock")]

    logging.info("FINISHED")
