#!/usr/bin/env python
# coding: utf-8

# # GSCV Model building

# ### Import Modules

# In[ ]:


import sys
import time
import joblib
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import randint


# In[ ]:


print(np.__version__)


# In[ ]:


import sklearn

sklearn.show_versions()


# ### Define Paths

# In[ ]:


# sys.path.clear()

# Basepath
basepath = "../"  # Project directory
sys.path.append(basepath)

# Data
DATA_PATH = basepath + "data"

# Results path
RESULTS_PATH = basepath + "03_Modelling/03_2_gscv/gscv_results/"

# Figure path
FIGURE_PATH = basepath + "03_Modelling/03_2_gscv/gscv_figures/"

# Path to environment
ENV_PATH = "/home/fhwn.ac.at/202375/.conda/envs/thesis/lib"

# Modelpath
MODEL_PATH = basepath + "models"

# Logging
LOG_DIR = basepath + "03_Modelling/03_2_gscv/"

# Active Learning library
AL_PATH = basepath + "al_lib"

# Add the paths
sys.path.extend(
    {DATA_PATH, FIGURE_PATH, ENV_PATH, MODEL_PATH, RESULTS_PATH, LOG_DIR, AL_PATH}
)
sys.path  # Check if the path is correct


# ### Logging

# In[ ]:


LOG_DIR


# In[ ]:


# import the logging specifications from file 'logging_config.py'
from al_lib.logging_config import create_logger
import datetime

# Add data/time information
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")

# Define the notebook name and the output name
notebook_name = "03_2_gscv.ipynb"  # Can also used when saving the notebook 

# Specify logging location
log_file_name = f"{notebook_name.split('.')[0]}_{date}.log"
log_file_path = f"{LOG_DIR}/{log_file_name}"

# Get the logger
logging = create_logger(__name__, log_file_path=log_file_path)

# Usage of the logger as follows:
logging.info("Logging started")


# # Import Data

# ## Import dpsDeriv1200.csv

# In[ ]:


data_dps_deriv_1200 = pd.read_csv(
    DATA_PATH + "/dpsDeriv1200.csv", sep=",", decimal=".", encoding="utf-8"
)
data_dps_deriv_1200 = data_dps_deriv_1200.rename(columns=lambda x: x.replace("X", ""))
data_dps_deriv_1200 = data_dps_deriv_1200.rename(columns={"Unnamed: 0": "Samplename"})
data_dps_deriv_1200


# ## Select Data

# In[ ]:


# Switch for the dataset
# Select from (data_small, data_full, data_2nd_deriv, data_dps_deriv_1200) or other if implemented
data_raw = data_dps_deriv_1200
data_raw.dataset_name = "data_dps_deriv_1200"
logging.info(f"Dataset: {data_raw.dataset_name}")
logging.info(f"Size of the dataset: {data_raw.shape}")


# ## Modelling Parameters

# In[ ]:


# Define the parameters for the CV

# Switch for testing mode (use only 10% of the data, among others)
testing = False

# Define a random state for randomized processes
random_state = np.random.RandomState(202375)

if testing == True:
    nfolds = 5
    NoTrials = 5
    n_jobs = 20
    save_model = False
    # data = data_raw.sample(frac=0.15, random_state=random_state)
    data = data_raw
    # logging.info(f"Size of the dataset reduced: {data.shape}")
else:
    nfolds = 10 # 10-fold cross-validation
    NoTrials = 15 # Number of trials
    n_jobs = 20
    save_model = True
    data = data_raw
    logging.info(f"Size of the dataset not reduced: {data.shape}")

# Log the modelling parameters
logging.info(
    f"Testing for Cross Validation: {testing}, nfolds: {nfolds}, NoTrials: {NoTrials}, n_jobs: {n_jobs}"
)


# ## Preprocessing
# 
# To apply the models we need to split the data into the variables and target.

# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


# Split into target and features
# The goal is to predict the year column of the dataset using the spectral data
X = data.select_dtypes("float")
y = data["year"]
X.shape, y.shape


# In[ ]:


# count the number of columns with std = 0.0 in X
logging.info(f"Number of columns dropped, where std = 0.0 in X: {(X.std() == 0.0).sum()}")


# In[ ]:


# drop the columns with std = 0.0
X = X.loc[:, X.std() != 0.0]
X.shape, y.shape
logging.info(f"Dimensions of X after dropping columns with std = 0.0: {X.shape}")
logging.info(f"Dimensions of Y: {y.shape}")


# In[ ]:


# Split the data into training and test sets
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)
logging.info(f"random split with testsize {test_size} into training and test sets")


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape
# assert the shapes and raise an error if they are not equal
assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


# ## Define Score metrics

# In[ ]:


from sklearn.metrics import make_scorer
from sklearn.metrics import root_mean_squared_error

# create a scorer which calculates Root Mean Squeared Error (RMSE)

scoring = make_scorer(root_mean_squared_error, greater_is_better=False)
# scoring = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
logging.info(f"Scorer: {scoring}")


# # Modeling with Grid Search Crossvalidation (GSCV)

# #### Models

# In[ ]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.neural_network import MLPRegressor as MLP
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.ensemble import RandomForestRegressor


# ## Hyperparameter Definition
# 
# Grid Search CV is usefull for the extensive testing of a defined parameter space. The results can consequently be used with confidence in their local validity. To create the local grid we will explore the space between the three most successful parameters in the rscv approach

# ### Calculate the parameters on basis of the rscv results
# 
# Of special intrest are the min and max value for numerical variables and the optimal value according to rscv. 
# For categorial variables, we  

# In[ ]:


RSCV_DIR = basepath + "03_Modelling/03_1_rscv/rscv_results/"


# In[ ]:


def _read_and_parse_csv(file_path):
    """
    Reads a CSV file containing RSCV results, parses the parameters,
    and returns a DataFrame with the parsed parameters.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    # drop rows where the entry in the 'params' column is 'params'
    df = df[df['params'] != 'params']
    # Parse the parameters
    df['params'] = df['params'].apply(eval)
    # split the params column into separate columns
    df_params = pd.DataFrame(df['params'].to_list(), index=df.index)
    df = df.drop(columns=['params'], axis = 1)
    # merge the dataframes
    df = pd.concat([df, df_params], axis=1)
    logging.info (f"Loaded and parsed {file_path} successfully.")
    # change the data types of the columns 'RMSE' and 'MAE' to float
    df['RMSE'] = df['RMSE'].astype('float')
    df['MAE'] = df['MAE'].astype('float')
    cl_types = df.dtypes

    for w,v in cl_types.items():
        logging.info(f"datatype of values in Column {w} : {v}")
    
    return df

# Example usage
pls_path = RSCV_DIR + 'pls_rscv_results.csv'
rscv_results_pls = _read_and_parse_csv(pls_path)


# In[ ]:


def _create_grid(rscv_results, grid_file = None):
    """
    Creates a grid from the RSCV results for GridSearchCV.
    It spans the hyperparameter space for the top 5 models, and includes 
    the specific values for the optimal model. 
    For categorical hyperparameters each unique entry is integrated. 
    """

    # transform the rscv_results into a dataframe

    # Select the 5 best performing models based on RMSE
    best_models = rscv_results.nsmallest(5, "RMSE")
    
    # Initialize a empty grid
    grid = {}
    model_name = str(rscv_results['model'].iloc[0]).removeprefix('<class').removesuffix('>').removesuffix("()")
    # Iterate over each column (parameter) except 'model', 'MAE', and 'RMSE'
    for col in rscv_results.columns.drop(["model", "MAE", "RMSE"]):
        # Extract the minimum and maximum values for numerical columns
        param_space = []
        if rscv_results[col].dtype in ['float64']:
            min_val = best_models[col].min()
            max_val = best_models[col].max()
            opt_val = best_models[col].iloc[0]
            # Create an equidistant grid around the min and max values
            param_space.append(np.linspace(min_val, max_val, 5))
            # if the optimal value is not in the grid, add it
            if opt_val not in param_space[-1]:
                param_space.append([opt_val])
        if rscv_results[col].dtype in ['int64']:
            min_val = best_models[col].min()
            max_val = best_models[col].max()
            opt_val = best_models[col].iloc[0]
            # Create an equidistant grid around the min and max values
            param_space.append(np.linspace(min_val, max_val, 5))
            if opt_val not in param_space[-1]:
                param_space.append([opt_val])
            #round the values and convert to int
            param_space = [np.round(val).astype(int) for val in param_space]
            # ensure there are no duplicates
        # For categorical columns, use all unique values
        else:
            unique_vals = best_models[col].unique()
            param_space.append(unique_vals)
        # Store the parameter space in the grid
            # key = column name 
            # value = parameter space
        param_space = [np.unique(val) for val in param_space]
        grid[col] = param_space[0]
    if grid_file is not None:
        # create the file
        with open(grid_file, 'w') as file:
            # write the gridname to the file 
            file.write(f"gscv_parameters = {str(grid)}")
            logging.info(f"Grid saved as {grid_file}")
            file.close()
    return grid


# In[195]:


# test the grid extraction methods for the pls

def _test_grid(): 
    """ Test the grid extraction method for PLS.
    Returns 'None' and a logging entry if the test is successful.
    """
    grid_pls = _create_grid(rscv_results_pls)
    # instantiate the pls model with the first entry in the grid dict
    mock_parms = {key: value[0] for key, value in grid_pls.items()}
    # create a PLS model
    try: 
        PLSRegression(**mock_parms)
        logging.info(f"PLS model instantiated successfully.")
    except:
        logging.error(f"PLS model could not be instantiated.")
    return grid_pls

_test_grid()


# ## Changed to markdown
# 
# models = {
#     "rf": RandomForestRegressor(),
#     "pls": PLSRegression(),
#     "krr": KRR(),
#     "mlp": MLP(),
#     "xgb": XGBRegressor(),
#     "hgb": HGB(),
# }
# 
# # Prepare objects to store the results
# for model in models.keys():
#     globals()[f"{model}_gscv_results"] = pd.DataFrame(
#         columns=["model", "MAE", "RMSE", "params"]
#     )
#     print(f"{model}_gscv_parameters successfully created.")

# In[ ]:


from al_lib.helper_functions import rmse_func as rmse
from al_lib.helper_functions import report_model
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.metrics import (
    mean_squared_error,
)  # also imports the neg_root_mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import make_scorer

def gscv(
    features,
    target,
    model,
    param_grid,
    results_file,
    NoTrials=5,
    nfolds=4,
    n_jobs=5,
    scoring=scoring, 
):
    """_summary_

    Args:
        features (_type_): _description_
        target (_type_): _description_
        model (_type_): _description_
        param_distributions (_type_): _description_
        results_file (_type_): _description_
        random_state (_type_): _description_
        NoTrials (int, optional): _description_. Defaults to 5.
        nfolds (int, optional): _description_. Defaults to 4.
        n_jobs (int, optional): _description_. Defaults to 5.
        scoring (_type_, optional): _description_. Defaults to scoring.

    Returns:
        _type_: _description_
    """
    # log the args
    logging.info(
        f"Features: {features.shape}, Target: {target.shape}, Model: {model}, Param_distributions: {param_grid}, Results File: {results_file} Random_state: {random_state}, NoTrials: {NoTrials}, nfolds: {nfolds}, n_jobs: {n_jobs}, Scoring: {scoring}"
    )

    # prepare the result object
    gscv_results = pd.DataFrame(columns=["model", "MAE", "RMSE", "params"])

    # define the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=random_state
    )
    # create the result objects 2
    gscv_rmse_inner = np.zeros(NoTrials)
    gscv_rmse_outer = np.zeros(NoTrials)

    for i in range(NoTrials):
        logging.info(f"Trial: {i} out of {NoTrials}")
        # split for nested cross-validation
        inner_cv = KFold(n_splits=nfolds, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=nfolds, shuffle=True, random_state=i)

        # non-nested parameter search and scoring
        gscv = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
                            )

        # fit
        gscv.fit(X_train, y_train)
        # make predictions to later estimate the generalization error
        y_pred = cvp(gscv, X_test, y_test, cv=outer_cv, n_jobs=n_jobs)
        all_predictions = np.zeros((len(y_test), NoTrials))
        all_predictions[:, i] = y_pred
        # calculate the RMSE for the inner and outer CV
        gscv_rmse_inner[i] = gscv.best_score_
        # calculate the RMSE for the outer CV
        gscv_rmse_outer[i] = rmse(y_test, y_pred)
        # store the results
        gscv_results.loc[i, "model"] = gscv.estimator
        gscv_results.loc[i, "MAE"] = mean_absolute_error(y_test, y_pred)
        gscv_results.loc[i, "RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        gscv_results.at[i, "params"] = gscv.best_params_
        report_model(gscv)

    # write results into outputifle
    gscv_results.to_csv(results_file, index=False, mode="a")

    return gscv_results
    # the goal of the rscv is to find the optimal hyperparameters
    # for further investigation we want to store
    # the 10 best model parameters and their scores
    # both the inner and outer cv scores, as well as the score difference


# # Random Forest Regressor - GSCV

# In[ ]:


rf = RandomForestRegressor()
# Import parameters
rf_path = RSCV_DIR + "rf_rscv_results.csv"
rscv_results_rf = _read_and_parse_csv(rf_path)
# save the grid as a txt.file
grid_file_rf = RESULTS_PATH + "rf_grid.txt"
grid_rf = _create_grid(rscv_results_rf, grid_file_rf)
logging.info(f"Grid for RF {grid_rf}")

# Define the results file
rf_gscv_results_file = f"{RESULTS_PATH}rf_gscv_results.csv"
logging.info(f"Results file: {rf_gscv_results_file}")


# In[ ]:


# performing GSCV for RF

rf_gscv_results = gscv(
    features = X,
    target = y,
    model = rf, 
    param_grid = grid_rf,
    results_file = rf_gscv_results_file,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs,
    scoring=scoring, 
)


# In[ ]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
rf_results = pd.read_csv(rf_gscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_rf = rf_results.loc[rf_results["RMSE"].idxmin(), "params"]
optimal_params_rf = dict(eval(optimal_params_str_rf))

rf_opt = RandomForestRegressor(**optimal_params_rf, random_state=random_state)

y_pred_rf = rf_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_rf.replace("np.int64", "").replace("(", "").replace(")", "")

title_str = (
    f"Random Forest: Actual vs. Predicted Values \n params:"
    + title_str_params
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_rf):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_rf.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 8)
rf_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_rf, param_dict, fig_path)


# # PLS Regressor - GSCV

# In[ ]:


pls = PLSRegression()
# Import parameters
pls_path = RSCV_DIR + "pls_rscv_results.csv"
rscv_results_pls = _read_and_parse_csv(pls_path)
# save the grid as a txt.file
grid_file_pls = RESULTS_PATH + "pls_grid.txt"
grid_pls = _create_grid(rscv_results_pls, grid_file_pls)
logging.info(f"Grid for PLS: {grid_pls}")

# Define the results file
pls_gscv_results_file = f"{RESULTS_PATH}pls_gscv_results.csv"
logging.info(f"Results file: {pls_gscv_results_file}")


# In[ ]:


pls_gscv_results = gscv(
    features = X,
    target = y,
    model = pls, 
    param_grid = grid_pls,
    results_file = pls_gscv_results_file,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs,
    scoring=scoring, 
)


# In[ ]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
pls_results = pd.read_csv(pls_gscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_pls = pls_results.loc[pls_results["RMSE"].idxmin(), "params"]
optimal_params_pls = dict(eval(optimal_params_str_pls))

pls_opt = PLSRegression(**optimal_params_pls)

y_pred_pls = pls_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_pls.replace("np.int64", "").replace("(", "").replace(")", "")

title_str = (
    f"PLS Regression: Actual vs. Predicted Values \n params:"
    + title_str_params
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_pls):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_pls.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 8)
rf_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_pls, param_dict, fig_path)


# # KRR - GSCV

# In[ ]:


from sklearn.kernel_ridge import KernelRidge as KRR
krr = KRR()
# Import parameters
krr_path = RSCV_DIR + 'krr_rscv_results.csv'
rscv_results_krr = _read_and_parse_csv(krr_path)
# save the grid as a txt.file
grid_file_krr = RESULTS_PATH + "krr_grid.txt"
grid_krr = _create_grid(rscv_results_krr, grid_file_krr)
logging.info(f"Grid for krr: {grid_krr}")

# Define the results file
krr_gscv_results_file = f"{RESULTS_PATH}krr_gscv_results.csv"
logging.info(f"Results file: {krr_gscv_results_file}")


# In[ ]:


krr_gscv_results = gscv(
    features = X,
    target = y,
    model = krr, 
    param_grid = grid_krr,
    results_file = krr_gscv_results_file,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs,
    scoring=scoring, 
)


# In[ ]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
krr_results = pd.read_csv(krr_gscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_krr = krr_results.loc[krr_results["RMSE"].idxmin(), "params"]
optimal_params_krr = dict(eval(optimal_params_str_krr))

krr_opt = KRR(**optimal_params_krr)

y_pred_krr = krr_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_krr.replace("np.int64", "").replace("(", "").replace(")", "").replace("np.float64", "")

title_str = (
    f"Kernel-Ridge Regression: Actual vs. Predicted Values \n params:"
    + title_str_params
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_krr):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_krr.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 8)
krr_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_krr, param_dict, fig_path)


# # mlp Regressor - GSCV

# In[ ]:


# import mlp
from sklearn.neural_network import MLPRegressor as MLP

# Import parameters
mlp_path = RSCV_DIR + 'mlp_rscv_results.csv'
rscv_results_mlp = _read_and_parse_csv(mlp_path)

# handle the NaN values in the mlp results['max_iter'] column
    # replace the NaN values with the median of the column
rscv_results_mlp['max_iter'] = rscv_results_mlp['max_iter'].fillna(rscv_results_mlp['max_iter'].median())
    # change the data type of the column to int
rscv_results_mlp['max_iter'] = rscv_results_mlp['max_iter'].astype(int)

# save the grid as a txt.file
grid_file_mlp = RESULTS_PATH + "mlp_grid.txt"
grid_mlp = _create_grid(rscv_results_mlp, grid_file_mlp)
logging.info(f"Grid for mlp {grid_mlp}")


# In[ ]:


# import mlp
from sklearn.neural_network import MLPRegressor as MLP

mlp = MLP()

mlp_gscv_results_file = f"{RESULTS_PATH}/mlp_gscv_results.csv"
logging.info(f"Results file: {mlp_gscv_results_file}")

mlp_gscv_results = gscv(
    features = X,
    target = y,
    model = mlp, 
    param_grid = grid_mlp,
    results_file = mlp_gscv_results_file,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs,
    scoring=scoring, 
)


# In[ ]:


# generate the actual vs. predicted plot

# import parameters
mlp_results = pd.read_csv(mlp_gscv_results_file)
optimal_params_str_mlp = mlp_results.loc[mlp_results["RMSE"].idxmin(), "params"]
optimal_params_mlp = dict(eval(optimal_params_str_mlp))

mlp_opt = MLP(**optimal_params_mlp)

y_pred_mlp = mlp_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_mlp.replace("np.int64", "").replace("(", "").replace(")", "").replace("np.float64", "")

title_str = (
    f"MLP: Actual vs. Predicted Values \n params:"
    + title_str_params
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_mlp):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_mlp.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 8)
mlp_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_mlp, param_dict, fig_path)


# # HGB - GSCV

# In[ ]:


# HGB
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

hgb_path = RSCV_DIR + 'hgb_rscv_results.csv'
rscv_results_hgb = _read_and_parse_csv(hgb_path)

# save the grid as a txt.file
grid_file_hgb = RESULTS_PATH + "hgb_grid.txt"
grid_hgb = _create_grid(rscv_results_hgb, grid_file_hgb)
logging.info(f"Grid for hgb {grid_hgb}")


# In[ ]:


hgb = HGB()

hgb_gscv_results_file = f"{RESULTS_PATH}/hgb_gscv_results.csv"
logging.info(f"Results file: {hgb_gscv_results_file}")

hgb_gscv_results = gscv(
    features = X,
    target = y,
    model = hgb, 
    param_grid = grid_hgb,
    results_file = hgb_gscv_results_file,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs,
    scoring=scoring, 
)


# In[ ]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
hgb_results = pd.read_csv(hgb_gscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_hgb = hgb_results.loc[hgb_results["RMSE"].idxmin(), "params"]
optimal_params_hgb = dict(eval(optimal_params_str_hgb))

hgb_opt = hgb(**optimal_params_hgb)

y_pred_hgb = hgb_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_hgb.replace("np.int64", "").replace("(", "").replace(")", "").replace("np.float64", "")

title_str = (
    f"Kernel-Ridge Regression: Actual vs. Predicted Values \n params:"
    + title_str_params
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_hgb):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_hgb.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 8)
hgb_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_hgb, param_dict, fig_path)


# # XGB - GSCV

# In[ ]:


# import xgboost
from xgboost import XGBRegressor

xgb = XGBRegressor()

# Import the XGB results
xgb_path = RSCV_DIR + 'xgb_rscv_results.csv'
rscv_results_xgb = _read_and_parse_csv(xgb_path)
#save the grid as a txt.file
grid_file_xgb = RESULTS_PATH + "xgb_grid.txt"

grid_xgb = _create_grid(rscv_results_xgb, grid_file_xgb)
logging.info(f"Grid for XGB: {grid_xgb}")


# In[ ]:


xgb_gscv_results_file = f"{RESULTS_PATH}/xgb_gscv_results.csv"
logging.info(f"Results file: {xgb_gscv_results_file}")

xgb_gscv_results = gscv(
    features = X,
    target = y,
    model = xgb, 
    param_grid = grid_xgb,
    results_file = xgb_gscv_results_file,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs,
    scoring=scoring, 
)


# In[ ]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
xgb_results = pd.read_csv(xgb_rscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_xgb = xgb_results.loc[xgb_results["RMSE"].idxmin(), "params"]
optimal_params_xgb = dict(eval(optimal_params_str_xgb))

xgb_opt = xgb(**optimal_params_xgb)

y_pred_xgb = xgb_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_xgb.replace("np.int64", "").replace("(", "").replace(")", "").replace("np.float64", "")

title_str = (
    f"Kernel-Ridge Regression: Actual vs. Predicted Values \n params:"
    + title_str_params
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_xgb):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_xgb.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 8)
xgb_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_xgb, param_dict, fig_path)


# # Quality Control
# 
# In this section the goal is to document the packages which where used during the execution of this notebook

# In[ ]:


## Package informations
from sklearn import show_versions

show_versions()

