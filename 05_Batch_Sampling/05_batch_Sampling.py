#!/usr/bin/env python
# coding: utf-8

# This notebook implements the active Learning strategies in the batch sampling setting, while using an oop approach. The goal of this notebook is to
# 
# * implement batch sampling for the KRR Model
# 

# # Setup
# ## Define the PATHS

# In[ ]:


import sys

basepath = "../"  # Project directory
sys.path.append(basepath)
# AL Scripts
AL_SCRIPTS_PATH = basepath + "al_lib/"

sys.path.append({AL_SCRIPTS_PATH})

from al_lib.active_learning_setting import ActiveLearningBatchSamplingPaths

PATHS = ActiveLearningBatchSamplingPaths()
(DATA_PATH, FIGURE_PATH, ENV_PATH, RESULTS_PATH, LOG_DIR) = PATHS

sys.path.extend(PATHS)

sys.path


# ## Include a logger

# In[ ]:


# import the logging specifications from file 'logging_config.py'
from al_lib.logging_config import create_logger
import datetime

# Add data/time information
date = datetime.datetime.now().strftime("%Y-%m-%d")
# date = now.strftime("%Y-%m-%d")
log_file_name = f"{date}_active_learning_batch.log"
log_file_path = f"{LOG_DIR}{log_file_name}"

# Create logger
logging = create_logger(__name__, log_file_path = log_file_path)
# Usage of the logger as follows:
logging.info("Logging started")
logging.info(f"log stored at: {log_file_path}")


# # Imports
# ## Packages

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge as KRR
import pandas as pd
import time


# ### sklearn warnings

# In[ ]:


## Turn of sklearn warnings
from warnings import simplefilter
import warnings

from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings(
    "ignore", message=".*y residual is constant*.", category=UserWarning, append=False
)
# logging.warning("Warning \"y residual is constant\" turned off")


# ### Import Data

# In[ ]:


# Define the datafile

data_name = "dpsDeriv1200.csv"

datafile = DATA_PATH + data_name

from al_lib.helper_functions import import_dpsDeriv1200
import pandas as pd

data = import_dpsDeriv1200(datafile)
logging.info(f"Data loaded and preprocessed from {datafile}")


# ## Import Timer

# In[ ]:


import time

start = time.time()
for i in range(100):
    pass
end = time.time()
print(f"Time taken for 100 passes: {end-start}")


# ## Split into feature and target variables

# In[ ]:


X = data.select_dtypes("float")
y = data["year"]
X.shape, y.shape


# ## Validation
# 
# since not every regression method is able to estimate its prediction accuracy, a split of the data is retained as validation set. 

# In[ ]:


# count the number of columns with std = 0.0 in X
logging.info(
    f"{(X.std() == 0.0).sum()} Columns dropped, where std = 0.0 in X"
)

# drop the columns with std = 0.0
X = X.loc[:, X.std() != 0.0]
logging.info(
    f"X: {X.shape},y: {y.shape} Dimensions after dropping columns with std = 0.0"
)


# # Train/Test/Validation Split

# In[ ]:


# Computational Settings
random_state = 12345

validation_size = 0.1
test_size = 0.3

from sklearn.model_selection import train_test_split

# retain 10% of the data for validation
(
    X_remainder,
    X_val,
    y_remainder,
    y_val,
) = train_test_split(X, y, test_size=validation_size, random_state=random_state)

# split the remainder into training and test (30%) set
X_train, X_test, y_train, y_test = train_test_split(
    X_remainder, y_remainder, test_size=test_size, random_state=random_state
)
logging.info(f"Split of the dataset into Train/Test/Validation set")

# assert the shapes for the sets and raise an error if they are not equal
assert X_train.shape[0] + X_test.shape[0] + X_val.shape[0] == X.shape[0], "Sum of samples in Train/Test/Validation set not equal to total samples"
assert X_train.shape[1] == X_test.shape[1] == X_val.shape[1] == X.shape[1], "Number of features in Train/Test/Validation set not equal to total features"
assert y_train.shape[0] + y_test.shape[0] + y_val.shape[0] == y.shape[0], "Sum of Sample-targets in Train/Test/Validation set not equal to total samples"
assert X_train.shape[0] == y_train.shape[0], "Number of samples not equal to number of targets in Train set"
assert X_test.shape[0] == y_test.shape[0], "Number of samples not equal to number of targets in Test set"
assert X_val.shape[0] == y_val.shape[0], "Number of samples not equal to number of targets in Validation set"

logging.info(f"Shapes of Train/Test/Validation set verified")
logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")


# In[ ]:


from al_lib.helper_functions import calc_set_sizes

calc_set_sizes(X_train , X_test, X_val, logging)


# # Model Parameters and Model Methods
# 
# The optimal model parameters according to the CV Results are used to to fit the individual models. 

# In[ ]:


# Import the Regressors

from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.cross_decomposition import PLSRegression as PLS
from xgboost import XGBRegressor as XGB

# Define the regressors
Regressors = [KRR]


# In[ ]:


# import the rscv model parameters from 03_Modelling/03_1_rscv

rscv_results_dir = basepath + "03_Modelling/03_1_rscv/rscv_results/"
gscv_results_dir = basepath + "03_Modelling/03_2_gscv/gscv_results/"

# models tested
models = ["hgb", "krr", "mlp", "pls", "rf", "xgb"]
from al_lib.helper_functions import _get_optimal_params_from_cv

optimal_params, rmse_from_cv, models_available = _get_optimal_params_from_cv(models, gscv_results_dir, rscv_results_dir, logging=logging)


# Return the model parameters which are used to perform active learning. These are important, since they potentially influence the performance of the AL-processes in an influental manner.

# In[ ]:


for model in models_available:
    if models_available[model]:
        logging.info(f"Optimal parameters (GSCV) for {model} with RMSE {rmse_from_cv[model]}: {optimal_params[model]}")
    else:
        logging.info(f"Optimal parameters (RSCV) for {model} with RMSE {rmse_from_cv[model]}: {optimal_params[model]}")


# In[ ]:


# generate object with the optimized parameters to hand over to the Regressors
for key in optimal_params.keys():
    # generate a global variable with the optimal parameters
    globals()[f"params_{key}"] = optimal_params[key]


for key in optimal_params.keys():
    logging.info(f"Optimal parameters for {key}: {optimal_params[key]}")


# # Active Learning Setup
# 
# The Basic Active Learning Experiment follows the specifications:
# 
# * Implementation of each Sampling Strategy in a Modular fashion
# * Selecting the inital samples randomly
# * Refitting the model after each selected sample
# * Runing the experiment n-fold with differing random states
# * Calculation of mean performance with confidence intervalls
# * Visualizing the results

# # Importing the sampling strategies

# In[ ]:


from al_lib.selection_criteria import (_random_selection,
                                       _gsx_selection, 
                                       _gsy_selection, 
                                       _uncertainty_selection, 
                                       _distance_weighing)


# # Defining the active Learning Framework

# In[ ]:


from al_lib.helper_functions import _validate_parameters
from al_lib.helper_functions import _rnd_initial_sampling
from al_lib.helper_functions import rmse_func
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

def active_learning_batch(
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    logging,
    model_class=None,
    model_params={},
    selection_criterion=None,
    n_iterations=None,
    n_samples_per_it=None,
    init_sample_size=None,
    random_state=None,
    n_jobs=None,
    results_file=None,
    **kwargs,
):
    """
    Perform active learning with the given parameters. 
    
    Active Learning selects additional samples for the training set from 
    provided pool of samples. The selection is based on a selection criterion, 
    which can be selected from implemented criteria. The active 
    learning process is repeated for n_iterations. The model is retrained after
    each iteration with the updated training set.

    Parameters
    ----------
    X_i : pd.DataFrame
        The features of the set (i = (train, test, val))
    y_i : pd.Series
        The target of the set (i = (train, test, val))
    model_class : model class, either from sklearn or xgboost
        The model to be used for the active learning process
    model_params : dict
        The parameters for the model
    selection_criterion : function
        The selection criterion to be used for the active learning process
    n_iterations : int, optional (default=50)
        The number of iterations for the active learning process
    n_samples_per_it : int, optional (default=1)
        The number of samples to be selected in each iteration
    init_sample_size : int, optional (default=10)
        The initial sample size for initial model
    random_state : int, optional
        The random state for the active learning process
    n_jobs : int, optional
        The number of kernels to be used for the active learning process
    results_file : str, optional
        The path to the file to store the results of the active learning process
        If provided, the results are stored as a csv file from a pandas dataframe
    **kwargs : dict
        Additional keyword arguments to be used for the selection criterion
    
    Returns
    -------
    tuple containing:
        rmse_test : np.array
            The RMSE of the model with the training set after each iteration
        rmse_validation : np.array
            The RMSE of the model with the validation set after each iteration
        samples_selected : np.array
            The samples selected in each iteration
        rmse_full : float
            The RMSE of the model trained with all training samples

    """
    if 'n_fold' in kwargs:
        n_fold = kwargs['n_fold']
    else:
        n_fold = 3 # default value

    # _validate_parameters(
    #     X_train,
    #     y_train,
    #     model_class=None,
    #     model_params={},
    #     selection_criterion=None,
    #     n_iterations=None,
    #     n_samples_per_it=None,
    #     init_sample_size=None,
    # )

    logging.info(f"Size of X_train: {X_train.shape}")
    logging.info(f"Size of y_train: {y_train.shape}")
    logging.info(f"Model_class: {model_class}")
    logging.info(f"Modelling parameters: {model_params}")
    logging.info(f"Selection Criterion: {selection_criterion}")
    logging.info(f"Key word arguments: {kwargs}")

    if n_samples_per_it is None:
        n_samples_per_it = 1
    if init_sample_size is None:
        init_sample_size = 10
    if n_iterations is None:
        n_iterations = 50
    if random_state is None:
        random_state = 12345

    # Initialize the model
    model = model_class(**model_params)
    # Initialize the active learning model
    X_Pool = X_train
    y_Pool = y_train

    # prepare the output objects
    rmse_test = np.zeros(n_iterations)
    rmse_validation = np.zeros(n_iterations)
    samples_selected = np.zeros(n_iterations)
    selection_value_storage = np.zeros(n_iterations)

    # initialize the learned set as a empty dataframe
    X_Learned = pd.DataFrame()
    y_Learned = pd.Series()
    
    # Initialize the model
    X_Learned, y_Learned, X_Pool, y_Pool = _rnd_initial_sampling(X_Pool, X_Learned, y_Pool, y_Learned, init_sample_size, random_state)
    model.fit(X_Learned, y_Learned)

    logging.info(f"Initial model fitted with {init_sample_size} samples")
    logging.info("--Active Learning starts--")

    for it in range(n_iterations):
        logging.info(f"Active Learning with {selection_criterion} - iteration: {it}")

        y_pred_pool = model.predict(X_Pool)
        # convert y_pred_pool to a 
        y_pred_pool = pd.Series(y_pred_pool, index=X_Pool.index)

        # uncertainty_criteria_(X_Pool, y_Pool, X_Learned, y_Learned, y_pred_pool, n_fold)
        sample_id, selection_value = selection_criterion(X_Pool = X_Pool, 
                                        y_Pool = y_Pool,
                                        X_Learned = X_Learned,
                                        y_Learned = y_Learned,
                                        y_pred_pool = y_pred_pool,
                                        n_fold = n_fold, 
                                        random_state = random_state, 
                                        logging = logging, 
                                        model = model,
                                        n_jobs = n_jobs,
                                        kwargs = kwargs)

        samples_selected[it] = sample_id
        selection_value_storage[it] = selection_value
        logging.info(f"Sample_id: {sample_id} with selection value {selection_value}")

        # Update the Sample sets
        x_new = X_Pool.loc[[sample_id]]
        y_new = y_Pool.loc[[sample_id]]
        X_Learned = pd.concat([X_Learned, x_new], ignore_index=True)
        y_Learned = pd.concat([y_Learned, y_new], ignore_index=True)
        X_Pool = X_Pool.drop(index=sample_id)
        y_Pool = y_Pool.drop(index=sample_id)

        # Update the Model
        # retrain model on the new full data set and predict a new fit, if the n_samples_per_it is reached
        if n_samples_per_it == None or n_samples_per_it == 1:
            model.fit(X_Learned, y_Learned)
            y_pred = model.predict(X_test)
            rmse_test[it] = rmse_func(y_test, y_pred)
            y_pred_val = model.predict(X_val)
            rmse_validation[it] = rmse_func(y_val, y_pred_val)
        if it % n_samples_per_it == 0 or it != 0:
            model.fit(X_Learned, y_Learned)
            y_pred = model.predict(X_test)
            rmse_test[it] = rmse_func(y_test, y_pred)
            y_pred_val = model.predict(X_val)
            rmse_validation[it] = rmse_func(y_val, y_pred_val)

    # write results into outputifle
    if results_file is not None:
        results = pd.DataFrame(
            {
                "rmse_test": rmse_test,
                "rmse_validation": rmse_validation,
                "samples_selected": samples_selected,
                "selection_value": selection_value_storage,
            }
        )
        results.to_csv(results_file, index=False, mode = "a")
        logging.info(f"Results written to {results_file}")

    # calc the rmse for the model including all training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_full = rmse_func(y_test, y_pred)

    # # plot the rmse over the iterations
    # plt.plot(range(n_iterations)[1:], rmse_test[1:])
    # # add a line for the model with all training samples
    # plt.axhline(y=rmse_full, color="r", linestyle="--")
    # selection_criterion_str = str(selection_criterion).split(" ")[1]
    # plt.title(
    #     f"RMSE over Iterations with {model_class} and\n {selection_criterion} as selection criterion \n {selection_criterion_str} as selection criterion"
    # )
    # plt.xlabel("Iteration")
    # plt.ylabel("RMSE")
    # plt.show()
    return rmse_test, rmse_validation, samples_selected, selection_value_storage, rmse_full


# In[ ]:


def test_active_learning_batch():
    """
    Function to test the active learning function
    """
    from al_lib.helper_functions import _create_test_data
    from al_lib.helper_functions import _test_params_krr

    # create test data
    X_train, X_test, X_val, y_train, y_test, y_val = _create_test_data(logging=logging)
    # Perform active Learning for n_iterations
    n_iterations = 5
    n_samples_per_it = 3
    initial_sample_size = 10
    model_params=_test_params_krr()
    model_class=KRR
    # Perform active learning
    rmse_test, rmse_validation, samples_selected, selection_value_storage, rmse_full = active_learning_batch(
        X_train,
        X_test,
        X_val,
        y_train,
        y_test,
        y_val,
        logging,
        model_class=model_class,
        model_params=model_params,
        selection_criterion=_uncertainty_selection,
        n_samples_per_it=n_samples_per_it,
        n_iterations=n_iterations,
        init_sample_size=initial_sample_size,
    )

    return rmse_test, rmse_validation, samples_selected, selection_value_storage, rmse_full

rmse_test, rmse_validation, samples_selected, selection_value_storage, rmse_full = test_active_learning_batch()

assert rmse_test.shape[0] == rmse_validation.shape[0] == samples_selected.shape[0], "Shapes of output arrays not equal"


# In[ ]:


# perform active learning twice and generate result plots

def test_active_learning_twice():
    """
    Function to test the active learning function
    """
    from al_lib.helper_functions import _create_test_data
    from al_lib.helper_functions import _test_params_krr

    # create test data
    X_train, X_test, X_val, y_train, y_test, y_val = _create_test_data(logging=logging)
    # Perform active Learning for n_iterations
    n_iterations = 15
    n_samples_per_it = 2
    initial_sample_size = 10
    model_params=_test_params_krr()
    model_class=KRR
    # Perform active learning using a loop, store the results for each iteration

    # the rsults will be stored in a df with the following columns:
    # for each n_al_iterations: rmse_test_(i), rmse_validation_(i), samples_selected_(i), rmse_full_(i)
    results = pd.DataFrame()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    for i in range(2):
        rmse_test, rmse_validation, samples_selected, selection_value_storage, rmse_full = active_learning_batch(
            X_train,
            X_test,
            X_val,
            y_train,
            y_test,
            y_val,
            logging,
            model_class=model_class,
            model_params=model_params,
            selection_criterion=_uncertainty_selection,
            n_samples_per_it=n_samples_per_it,
            n_iterations=n_iterations,
            init_sample_size=initial_sample_size,
        )
        results[f"rmse_test_{i}"] = rmse_test
        results[f"rmse_validation_{i}"] = rmse_validation
        results[f"samples_selected_{i}"] = samples_selected
        results[f"rmse_full_{i}"] = rmse_full
        results[f"selection_value_{i}"] = selection_value_storage

    # plot the results of both iterations
    # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # for i in range(2):
    #     ax[i].plot(range(n_iterations), results[f"rmse_test_{i}"], label="RMSE Sampling")
    #     ax[i].plot(range(n_iterations), results[f"rmse_validation_{i}"], label="RMSE Validation")
    #     ax[i].axhline(y=results[f"rmse_full_{i}"][0], color="r", linestyle="--", label="RMSE Full")
    #     ax[i].set_title(f"RMSE over Iterations with KRR and Uncertainty Sampling - Iteration {i}")
    #     ax[i].set_xlabel("Iteration")
    #     ax[i].set_ylabel("RMSE")
    #     ax[i].legend()
    # plt.show()
    # plot the results of both iterations in a single plot
        
    
        ax1.plot(range(n_iterations), (results[f"rmse_test_{i}"]-i//10), label=f"RMSE Sampling {i}")
        ax1.plot(range(n_iterations), results[f"rmse_validation_{i}"], label=f"RMSE Validation {i}")
        ax1.axhline(y=results[f"rmse_full_{i}"][0], color="r", linestyle="--", label="RMSE Full")
        ax1.set_title(f"RMSE over Iterations with KRR and Uncertainty Sampling")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("RMSE")
        ax1.legend()

        # prepare the results to plot the selection values
        ax2.plot(range(n_iterations), results[f"selection_value_{i}"], label=f"Selection Value {i}")
        ax2.set_title(f"Selection Value for Uncertainty Sampling")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Selection Value")
        ax2.legend()
    # id the range for the y axis
    max_y_test = max([results[f"rmse_test_{i}"].max() for i in range(2)])
    max_y_val = max([results[f"rmse_validation_{i}"].max() for i in range(2)])
    max_y  = max(max_y_test, max_y_val)
    ax1.set_ylim(0, max_y+0.1)
    plt.show()

test_active_learning_twice()


# # Main Experiment
# 
# The main experiment compares the performance of the various selection strategies statistically. To this end each selection strategy is performed multiple times for each model-class. The results can be compared. 

# In[ ]:


# remove the model and the model parameters for the models that are not available
for model in models_available:
    if models_available[model] == False:
        # remove the model and the model parameters
        try: 
            #remove the model from the models list
            models.remove(model)
        except KeyError:
            logging.info(f"Error deleting the parameters for model: {model}")
        try:
            del globals()[f"{model}"]
        except KeyError:
            logging.info(f"Error deleting the model: {model}")


# In[ ]:


# Define the models
# remove the models that are not available
# models = ["hgb", "krr", "pls", "rf", "xgb"]
models = ["rf", "krr", "hgb", "pls", "xgb"]
models_list = [RF, KRR, HGB, PLS, XGB]

params = [params_rf, params_krr, params_hgb, params_pls, params_xgb]

model_params_list = [{model: param} for model, param in zip(models_list, params)]
model_params_list


# In[ ]:


for model in model_params_list:
    #seperate the model class and the parameters
    model_class = list(model.keys())[0]
    model_params = model[model_class]
    print(model_class, model_params)


# In[ ]:


# Main Experiment

AL_RESULTS_PATH = f"{RESULTS_PATH}al_batch_tables/"

# number of active learning runs
n_al_iterations = 10
# Define the number of iterations for each active learning run
n_iterations = 250

# Define the number of samples to be queried in each iteration
n_samples_per_it = 5

# Define the initial sample size
init_sample_size = 5

# Define the random state
random_state = 12345

# Define the number of jobs
n_jobs = 30

# Define the output object

selection_criteria = [
    {
        "criteria": _random_selection,
        "crit_name": "random",
        "kwargs": {},
    },
    {"criteria": _gsx_selection, "crit_name": "gsx", "kwargs": {}},
    {"criteria": _gsy_selection, "crit_name": "gsy", "kwargs": {}},
    {
        "criteria": _uncertainty_selection,
        "crit_name": "uncertainty",
        "kwargs": {"n_fold": 3},
    },
    {"criteria": _distance_weighing, "crit_name": "idw", "kwargs": {}},
]

# prepare the results dataframe
# for model in models:
#     for criteria in selection_criteria:
#         # create a global variable with the model and criterion
#         globals()[f"{model}_{criteria}"] = pd.DataFrame(
#             index=range(n_al_iterations), columns=range(n_iterations)
#         )

# perform the active learning process

for model in model_params_list:
    start = time.time()
    model_class = list(model.keys())[0]
    model_params = model[model_class]
    results = pd.DataFrame()
    logging.info(f"Current model: {model}")
    for i in range(n_al_iterations):
        logging.info(f"Active Learning iteration: {i}")
        random_state = random_state + i
        for criteria in selection_criteria:
            logging.info(
                f"Current criterion: {criteria['crit_name']} with kwargs: {criteria['kwargs']}"
            )
            # extract the model name
            model_name = str(model_class).split(".")[-1]
            selection_criteria_name = criteria["crit_name"]
            results_file = f"{AL_RESULTS_PATH}al_batch_{models_available}_{criteria['crit_name']}.csv"
            kwargs = criteria.get("kwargs", {})
            (
                rmse_test,
                rmse_validation,
                samples_selected,
                selection_value_storage,
                rmse_full,
            ) = active_learning_batch(
                X_train,
                X_test,
                X_val,
                y_train,
                y_test,
                y_val,
                logging=logging,
                model_class=model_class,
                model_params=model_params,
                selection_criterion=criteria["criteria"],
                n_iterations=n_iterations,
                n_samples_per_it=n_samples_per_it,
                init_sample_size=init_sample_size,
                random_state=random_state,
                n_jobs=n_jobs,
                results_file=None,
                **kwargs,
            )
            results[f"rmse_test_{model_name}_{criteria['crit_name']}_{i}"] = rmse_test
            results[f"rmse_val_{model_name}_{criteria['crit_name']}_{i}"] = (
                rmse_validation
            )
            results[f"sample_sel_{model_name}_{criteria['crit_name']}_{i}"] = (
                samples_selected
            )
            results[f"rmse_full_{model_name}_{criteria['crit_name']}_{i}"] = rmse_full
            results[f"selection_value_{model_name}_{criteria['crit_name']}_{i}"] = (
                selection_value_storage
            )
    # store the results in the global variables
    globals()[f"{model_name}_al_results"] = results
    globals()[f"{model_name}_results"] = results
    logging.info(f"Results stored in global variable: {model_name}_al_results")

    # write the results to a csv file
    results.to_csv(f"{AL_RESULTS_PATH}al_results_{model_name}.csv")
    end = time.time()
    logging.info(f"Time taken for active learning with {model_name}: {end-start}")


# # Visualize the results

# In[ ]:


# define the path to the data

from al_lib.results_vis import load_data  
from al_lib.results_vis import _seperate_results_test
from al_lib.results_vis import _seperate_results_val
from al_lib.results_vis import _plot_rmse


# In[ ]:


# Import Data

TABLES_PATH = RESULTS_PATH + "al_batch_tables/"
# load the data
filename_krr = "al_results_KernelRidge'>.csv"

data_krr = load_data(filename_krr, TABLES_PATH)

# rename the columns for better readability
for col in data_krr.columns:
    # rename the columns
    data_krr.rename(columns={col: col.replace("KernelRidge'>", "KRR")}, inplace=True)

data_krr.drop(columns=["Unnamed: 0"], inplace=True)
data_krr.head()


# In[ ]:


# Import Data

TABLES_PATH = RESULTS_PATH + "al_batch_tables/"
# load the data
filename_pls = "al_results_PLSRegression'>.csv"

data_pls = load_data(filename_pls, TABLES_PATH)

# rename the columns for better readability
for col in data_pls.columns:
    # rename the columns
    data_pls.rename(columns={col: col.replace("PLSRegression'>", "PLS")}, inplace=True)

data_pls.drop(columns=["Unnamed: 0"], inplace=True)
data_pls.head()


# In[ ]:


# Import Data
# load the data
filename_hgb = "al_results_HistGradientBoostingRegressor'>.csv"

data_hgb = load_data(filename_hgb, TABLES_PATH)

# rename the columns for better readability
for col in data_hgb.columns:
    # rename the columns
    data_hgb.rename(columns={col: col.replace("HistGradientBoostingRegressor'>", "HGB")}, inplace=True)

data_hgb.drop(columns=["Unnamed: 0"], inplace=True)
data_hgb.head()


# In[ ]:


# Import Data
# load the data
filename_rf = "al_results_RandomForestRegressor'>.csv"

data_rf = load_data(filename_rf, TABLES_PATH)

# rename the columns for better readability
for col in data_rf.columns:
    # rename the columns
    data_rf.rename(columns={col: col.replace("RandomForestRegressor'>", "RF")}, inplace=True)

data_rf.drop(columns=["Unnamed: 0"], inplace=True)
data_rf.head()


# In[ ]:


# Import Data
# load the data
filename_xgb = "al_results_XGBRegressor'>.csv"

data_xgb = load_data(filename_xgb, TABLES_PATH)

# rename the columns for better readability
for col in data_xgb.columns:
    # rename the columns
    data_xgb.rename(columns={col: col.replace("XGBRegressor'>", "XGB")}, inplace=True)

data_xgb.drop(columns=["Unnamed: 0"], inplace=True)
data_xgb.head()


# define functions to perform the necessary operations
# 
# _seperate_results_test (al_lib)
# _seperate_results_val (al_lib)
# _plot_rmse (al_lib)
# _calculate_auc
# _combined_auc_plot
# _plot_selection_value_development

# In[ ]:


# Only to necessary, if not the full notebook is run

selection_criteria = [
    {
        "criteria": _random_selection,
        "crit_name": "random",
        "kwargs": {},
    },  #'random_state': random_state}},
    {"criteria": _gsx_selection, "crit_name": "gsx", "kwargs": {}},
    {"criteria": _gsy_selection, "crit_name": "gsy", "kwargs": {}},
    {
        "criteria": _uncertainty_selection,
        "crit_name": "uncertainty",
        "kwargs": {"n_fold": 3},
    },{"criteria": _distance_weighing, "crit_name": "idw", "kwargs": {}},
]


# In[ ]:


# extract test RMSE

model_name = "KRR"
filepath = FIGURE_PATH + "test_rmse_krr.png"
sep_res_test = _seperate_results_test(data_krr, model_name = model_name)
plot = _plot_rmse(
    sep_res_test,
    selection_criteria=selection_criteria,
    model_name=model_name,
    title=f"Test RMSE for {model_name}",
)


# In[ ]:


# extract validation RMSE

model_name = "KRR"
filepath = FIGURE_PATH + "val_rmse_krr.png"
sep_res_val = _seperate_results_val(data_krr, model_name = model_name)
plot = _plot_rmse(
    sep_res_val,
    selection_criteria=selection_criteria,
    model_name=model_name,
    title=f"Validation RMSE for {model_name}",
)


# In[ ]:


def _calculate_auc(results):  # test_rmse
    """
    Calculate the Area under the curve for each selection criteria from the
    post-processed results
    Parameters:
    results: list of dataframes
    Returns:
    aucs: list of tuples (AUC, std_rmse_mean_per_iteration, AUC_mean, auc_std)
        each tuple contains the AUC, the standard deviation of the AUC, the mean of the AUC and the standard deviation of the RMSE
        AUC: float, the area under the curve
        std_rmse_mean_per_iteration: float, the standard deviation of the RMSE for each iteration
        AUC_mean: float, the mean of the AUC
        auc_std: float, the standard deviation of the AUC
    """
    aucs = []
    for i, rmse in enumerate(results):  # for each selection criteria

        rmse_mean_per_iteration = rmse.mean(axis=1)  # axis = 1 -> row wise

        # calc the std for each iteration
        std_rmse_mean_per_iteration = rmse.std(axis=1).round(3)

        # calculate the area under the curve
        auc = sum(rmse_mean_per_iteration).__round__(1)

        # calculate the mean of the AUC
        auc_mean = np.mean(rmse_mean_per_iteration).__round__(1)

        # calculate the standard deviation of the RMSE for each iteration
        auc_std = rmse_mean_per_iteration.std().__round__(1)
        aucs.append((auc, std_rmse_mean_per_iteration, auc_mean, auc_std))
    return aucs

def _plot_auc(aucs, model_name, filepath, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, auc in enumerate(aucs):
        ax.bar(
            selection_criteria[i]["crit_name"],
            auc[2],
            yerr=auc[3],
            label=selection_criteria[i]["crit_name"],
            capsize=5,
        )
    if title == None:
        title = f"AUC/iteration for {model_name}"
    else:
        title = title
    ax.set_title(title)
    ax.set_ylabel("AUC")
    if filepath != None:
        plt.savefig(filepath)
    plt.show()
    return fig, ax


# In[ ]:


aucs_krr = _calculate_auc(sep_res_val)


# In[ ]:


from al_lib.results_vis import combined_auc_plot
## main report function 
def report_al_batch_results(results, model_name, filepath , selection_criteria):
    test_rmse = _seperate_results_test(results=results, model_name=model_name)
    val_rmse = _seperate_results_val(results = results, model_name=model_name)
    # generate the test rmse plot
    _plot_rmse(
        test_rmse,
        selection_criteria=selection_criteria,
        model_name=model_name,
        title=f"Test RMSE for {model_name}",
        filepath=filepath + f"test_rmse_{model_name}.png",
    )
    # generate the validation rmse plot
    _plot_rmse(
        val_rmse,
        selection_criteria=selection_criteria,
        model_name=model_name,
        title=f"Validation RMSE for {model_name}",
        filepath=filepath + f"val_rmse_{model_name}.png",
    )

    # revised
    mean_auc_test = _calculate_auc(test_rmse)
    mean_auc_val = _calculate_auc(val_rmse)
    combined_auc_plot(
        mean_auc_test = mean_auc_test, mean_auc_val = mean_auc_val, model_name=model_name, filepath=filepath, selection_criteria=selection_criteria
    )


# In[ ]:


filepath = FIGURE_PATH 
report_al_batch_results(data_krr, model_name = "KRR", filepath = filepath, selection_criteria = selection_criteria)


# In[ ]:


report_al_batch_results(
    data_pls, model_name="PLS", filepath=filepath, selection_criteria=selection_criteria
)


# In[ ]:


report_al_batch_results(
    data_rf, model_name="RF", filepath=filepath, selection_criteria=selection_criteria
)


# In[ ]:


report_al_batch_results(
    data_xgb, model_name="XGB", filepath=filepath, selection_criteria=selection_criteria
)

