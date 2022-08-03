# %% Imports
import numpy as np
import pandas as pd
from pathlib import Path
from neighbors import estimate_performance
from neighbors.base import Base
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning
import warnings
from fit_all_models import prepare_data


# %% Paths
base_dir = Path("/Users/Esh/Documents/dartmouth/cosan/projects/collab_filter")
analysis_dir = base_dir / "analysis"
data_dir = base_dir / "data"

datasets = dict(
    iaps=pd.read_csv(data_dir / "iaps.csv"),
    moth=pd.read_csv(data_dir / "moth.csv"),
    decisions=pd.read_csv(data_dir / "decisions.csv"),
)

# %% Define custom class that wraps sklearn's MICE implementation
class MICE(Base):
    def __init__(
        self, data, mask=None, n_mask_items=None, verbose=True, random_state=None
    ):
        super().__init__(
            data, mask, n_mask_items, random_state=random_state, verbose=verbose
        )

    def fit(self, dilate_by_nsamples=None, **kwargs):
        # Call parent fit which acts as a guard for non-masked data
        super().fit()

        self.dilate_mask(n_samples=dilate_by_nsamples)

        # Store the mean because we'll use it in cases we can't make a prediction
        self.mean = self.masked_data.mean(skipna=True, axis=0)

        # Initialize imputer with linear regression estimator
        self.model = IterativeImputer(
            estimator=LinearRegression(),
            imputation_order="random",
            min_value=self.data.min().min(),
            max_value=self.data.max().max(),
            **kwargs,
        )
        # NO NEED TO WORRY ABOUT THIS IF WE TELL THE IMPUTER TO IMPUTE OVER SUBS!
        # When a column has all nans (e.g. no ratings for a user or item)
        # IterativeImputer silently drops those columns which changes its shape. See:
        # https://github.com/scikit-learn/scikit-learn/issues/16977
        #  https://github.com/scikit-learn/scikit-learn/issues/16426

        X = self.masked_data.to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _ = self.model.fit(X)
        self._predict(X)
        self.is_fit = True

    def _predict(self, X):
        predictions = self.model.transform(X)
        try:
            self.predictions = pd.DataFrame(
                predictions, index=self.data.index, columns=self.data.columns
            )
        except ValueError as e:
            print(e)
            predictions = np.full_like(self.data, np.nan)
            self.predictions = pd.DataFrame(
                predictions, index=self.data.index, columns=self.data.columns
            )


# %% Run MICE for each dataset and sparsity level
n_mask_items = np.arange(0.1, 1.0, 0.1)

for dataset_name, data in tqdm(
    datasets.items(), desc="Dataset", position=0, leave=True
):
    out_file = analysis_dir / f"{dataset_name}_user_model_comparison.csv"

    if dataset_name == "iaps":
        basic_emotions = ["anger", "sadness", "joy", "surprise", "fear", "disgust"]
        dimensions = list(
            filter(lambda x: x in basic_emotions, data.Appraisal_Dimension.unique())
        )
    elif dataset_name == "moth":
        dimensions = data.movie.unique()
    elif dataset_name == "decisions":
        dimensions = ["Prop_Returned"]

    for dimension in tqdm(dimensions, desc="Dimension", position=1, leave=False):
        ratings = prepare_data(data, dataset_name, dimension)

        for n in tqdm(n_mask_items, desc="Sparsity", position=2, leave=False):
            _, user_results = estimate_performance(
                MICE,
                ratings,
                n_mask_items=n,
                return_agg=False,
                return_full_performance=True,
                verbose=False,
                timeit=True,
            )

            user_results["dilation"] = 0
            user_results["train_size"] = 1 - n
            user_results["dimension"] = dimension
            user_results["algorithm"] = "MICE"
            user_results["n_factors"] = "all"
            user_results["max_factors"] = min(ratings.shape)

            user_results.to_csv(out_file, mode="a", index=False, header=False)

# %%
