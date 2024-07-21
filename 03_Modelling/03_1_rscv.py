#!/usr/bin/env python
# coding: utf-8

# # RSCV Model building

# ### Import Modules

# In[1]:


import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import randint


# In[2]:


print(np.__version__)


# In[3]:


import sklearn

sklearn.show_versions()


# ### Define Paths

# In[4]:


# sys.path.clear()

# Basepath
basepath = "../"  # Project directory
sys.path.append(basepath)

# Data
DATA_PATH = basepath + "data"

# Results path
RESULTS_PATH = basepath + "03_Modelling/03_1_rscv/rscv_results/"

# Figure path
FIGURE_PATH = basepath + "03_Modelling/03_1_rscv/rscv_figures/"

# Path to environment
ENV_PATH = "/home/fhwn.ac.at/202375/.conda/envs/thesis/lib"

# Modelpath
MODEL_PATH = basepath + "models"

# Logging
LOG_DIR = basepath + "03_Modelling/03_1_rscv/"

# Active Learning library
AL_PATH = basepath + "al_lib"

# Add the paths
sys.path.extend(
    {DATA_PATH, FIGURE_PATH, ENV_PATH, MODEL_PATH, RESULTS_PATH, LOG_DIR, AL_PATH}
)
sys.path  # Check if the path is correct


# ### Logging

# In[5]:


LOG_DIR


# In[6]:


# import the logging specifications from file 'logging_config.py'
from al_lib.logging_config import create_logger
import datetime

# Add data/time information
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")

# Define the notebook name and the output name
notebook_name = "03_1_rscv.ipynb"  # Is also used when saving the notebook
output_name = f"{notebook_name.split('.')[0]}_{date}.html"

# Specify logging location
log_file_name = f"{notebook_name.split('.')[0]}_{date}.log"
log_file_dir = f"{LOG_DIR}"
log_file_path = f"{LOG_DIR}/{log_file_name}"
# print(f"Log file path: {log_file_path}")

# Get the logger
# logger = None
logging = create_logger(__name__, log_file_path=log_file_path)

# Usage of the logger as follows:
logging.info("Logging started")


# ### Import Data

# #### Import PS20191107_2deriv_gegl.csv

# In[7]:


# Import 2nd_deriv

data_ps2019_2deriv = pd.read_csv(
    DATA_PATH + "/PS20191107_2deriv_gegl.csv",
    on_bad_lines="skip",
    sep=";",
    decimal=",",
    encoding="utf-8",
)

data_ps2019_2deriv = data_ps2019_2deriv.rename(columns={"Unnamed: 0": "Name"})

# Convert all columns of type 'object' to 'float' or 'int' if possible
for column in data_ps2019_2deriv.columns:
    # change datatype from the 'year' column to 'int
    if column == "year":
        data_ps2019_2deriv[column] = data_ps2019_2deriv[column].astype("int")
        print(f"'{column}' has been converted to 'int'.")
        # skip the rest of the loop
        continue
    try:
        data_ps2019_2deriv[column] = data_ps2019_2deriv[column].astype("float")
        # data_small.select_dtypes(include=['object']).astype('float')
    except ValueError:
        print(f"'{column}' could not be converted(ValueError). Continue with other column(s).")
    except TypeError:
        print(f"'{column}' could not be converted(TypeError). Continue with other column(s).")


# In[8]:


data_ps2019_2deriv.shape  # for quality control purposes


# ## Import dpsDeriv1200.csv

# In[9]:


data_dps_deriv_1200 = pd.read_csv(
    DATA_PATH + "/dpsDeriv1200.csv", sep=",", decimal=".", encoding="utf-8"
)
data_dps_deriv_1200 = data_dps_deriv_1200.rename(columns=lambda x: x.replace("X", ""))
data_dps_deriv_1200 = data_dps_deriv_1200.rename(columns={"Unnamed: 0": "Samplename"})
data_dps_deriv_1200


# ## Select Data

# In[10]:


# Switch for the dataset
# Select from (data_small, data_full, data_2nd_deriv) or other if implemented
data_raw = data_dps_deriv_1200
data_raw.dataset_name = "data_dps_deriv_1200"
logging.info(f"Dataset: {data_raw.dataset_name}")
logging.info(f"Size of the dataset: {data_raw.shape}")


# ## Modelling Parameters

# In[11]:


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
    logging.info(f"Size of the dataset reduced: {data.shape}")
else:
    nfolds = 10
    NoTrials = 15
    n_jobs = 30
    save_model = True
    data = data_raw
    logging.info(f"Size of the dataset not reduced: {data.shape}")

# Log the modelling parameters
logging.info(
    f"Testing for Cross Validation: {testing}, nfolds: {nfolds}, NoTrials: {NoTrials}, n_jobs: {n_jobs}"
)


# ## Sklearn Warnings
# 
# During testing, some warnings are not relevant, and to simplify the output results warning filter can be cosntructed in sklearn. 

# In[28]:


from warnings import simplefilter
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning

# Turn of sklearn warnings for failed fits 
if testing == True: 
    simplefilter("ignore", category=FitFailedWarning)
    simplefilter("ignore", category=ConvergenceWarning)
    warnings.filterwarnings(
    "ignore", message=".*y residual is constant*.", category=UserWarning, append=False
    )


# ## Preprocessing
# 
# To apply the models we need to split the data into the variables and target.

# In[12]:


data.dtypes.unique()


# In[13]:


data.dtypes.value_counts()


# In[14]:


data.info()


# In[15]:


data.describe()


# In[16]:


# Split into target and features
# The goal is to predict the year column of the dataset using the spectral data
X = data.select_dtypes("float")
y = data["year"]
X.shape, y.shape


# In[17]:


# count the number of columns with std = 0.0 in X
logging.info(f"Number of columns dropped, where std = 0.0 in X: {(X.std() == 0.0).sum()}")


# In[18]:


# drop the columns with std = 0.0
X = X.loc[:, X.std() != 0.0]
X.shape, y.shape
logging.info(f"Dimensions of X after dropping columns with std = 0.0: {X.shape}")
logging.info(f"Dimensions of Y: {y.shape}")


# In[19]:


# Split the data into training and test sets
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)
logging.info(f"random split with testsize {test_size} into training and test sets")


# In[20]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape
# assert the shapes and raise an error if they are not equal
assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


# ## Define Score metrics

# In[21]:


from sklearn.metrics import make_scorer
from sklearn.metrics import root_mean_squared_error

# create a scorer which calculates Root Mean Squeared Error (RMSE)

scoring = make_scorer(root_mean_squared_error, greater_is_better=False)
# scoring = make_scorer(mean_squared_error, greater_is_better=False, squared=False)
logging.info(f"Scorer: {scoring}")


# # Modeling with Randomized Search Crossvalidation (RSCV)

# #### Models

# In[22]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.neural_network import MLPRegressor as MLP
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.ensemble import RandomForestRegressor


# ## Hyperparameter Definition
# 
# Randomized Search CV is usefull for the efficient exploration of a large parameter space. The results can consequently be used to design a fine grid for the Grid Search CV

# In[23]:


# load the Hyperparameter distributions for the RandomizedSearchCV
from al_lib.rscv_parameters import (
    rf_rscv_parameters,
    pls_rscv_parameters,
    krr_rscv_parameters,
    mlp_rscv_parameters,
    xgb_rscv_parameters,
    hgb_rscv_parameters,
)

# to update the import without restarting the kernel, uncoment and modify the following line
# del <model>_parameters


# In[24]:


import pandas as pd
import numpy as np

models = {
    "rf": RandomForestRegressor(),
    "pls": PLSRegression(),
    "krr": KRR(),
    "mlp": MLP(),
    "xgb": XGBRegressor(),
    "hgb": HGB(),
}

# Prepare objects to store the results
# Template:
# rf_rscv_results = pd.DataFrame(columns=["model", "MAE", "RMSE", "params"])
for model in models.keys():
    globals()[f"{model}_rscv_results"] = pd.DataFrame(
        columns=["model", "MAE", "RMSE", "params"]
    )
    print(f"{model}_rscv_parameters")


# ## Defining the rscv process

# In[25]:


from al_lib.helper_functions import rmse_func as rmse
from al_lib.helper_functions import report_model
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.metrics import (
    mean_squared_error,
)  # also imports the neg_root_mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import make_scorer


# create a scorer which calculates Root Mean Squeared Error (RMSE)


def rscv(
    features,
    target,
    model,
    param_distributions,
    results_file,
    random_state,
    NoTrials=5,
    nfolds=4,
    n_jobs=5,
    scoring=scoring, #
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
    logging.info(f"STARTED the RandomizedSearchCV for {model} with {NoTrials} trials")
    # log the args
    logging.info(
        f"Features: {features.shape}, Target: {target.shape}, Model: {model}, Param_distributions: {param_distributions}, Results File: {results_file} Random_state: {random_state}, NoTrials: {NoTrials}, nfolds: {nfolds}, n_jobs: {n_jobs}, Scoring: {scoring}"
    )
    logging.info(f"Results file: {results_file}")

    # prepare the result object 1
    rscv_results = pd.DataFrame(columns=["model", "MAE", "RMSE", "params"])

    # define the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=random_state
    )
    # create the result objects 2
    rscv_rmse_inner = np.zeros(NoTrials)
    rscv_rmse_outer = np.zeros(NoTrials)

    for i in range(NoTrials):
        logging.info(f"Trial: {i} out of {NoTrials}")
        # split for nested cross-validation
        inner_cv = KFold(n_splits=nfolds, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=nfolds, shuffle=True, random_state=i)

        # non-nested parameter search and scoring
        rscv = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=10,
            cv=inner_cv,
            random_state=random_state,
            scoring=scoring,

            n_jobs=n_jobs,
        )

        # fit
        rscv.fit(X_train, y_train)
        # make predictions to later estimate the generalization error
        y_pred = cvp(rscv, X_test, y_test, cv=outer_cv, n_jobs=n_jobs)
        all_predictions = np.zeros((len(y_test), NoTrials))
        all_predictions[:, i] = y_pred
        # calculate the RMSE for the inner and outer CV
        rscv_rmse_inner[i] = rscv.best_score_

        # calculate the RMSE for the outer CV
        # rscv_rmse_outer[i] = np.sqrt(mean_squared_error(y_test, y_pred))
        rscv_rmse_outer[i] = rmse(y_test, y_pred)
        # store the results
        rscv_results.loc[i, "model"] = rscv.estimator
        rscv_results.loc[i, "MAE"] = mean_absolute_error(y_test, y_pred)
        rscv_results.loc[i, "RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        rscv_results.at[i, "params"] = rscv.best_params_
        report_model(rscv)

    # write results into outputifle
    rscv_results.to_csv(results_file, index=False, mode="a")
    logging.info(f"FINISHED the RandomizedSearchCV for {model} with {NoTrials} trials")
    return rscv_results
    # the goal of the rscv is to find the optimal hyperparameters
    # for further investigation we want to store
    # the 10 best model parameters and their scores
    # both the inner and outer cv scores, as well as the score difference


# # Random Forest Regressor - RSCV

# In[102]:


rf = RandomForestRegressor()
rf_rscv_results_file = f"{RESULTS_PATH}rf_rscv_results.csv"

rscv(
    features=X,
    target=y,
    model=rf,
    param_distributions=rf_rscv_parameters,
    results_file=rf_rscv_results_file,
    random_state=random_state,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs
)


# In[103]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
rf_results = pd.read_csv(rf_rscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_rf = rf_results.loc[rf_results["RMSE"].idxmin(), "params"]
optimal_params_rf = dict(eval(optimal_params_str_rf))

rf_opt = RandomForestRegressor(**optimal_params_rf)

y_pred_rf = rf_opt.fit(X_train, y_train).predict(X_test)

title_str = (
    f"Random Forest: Actual vs. Predicted Values \n params:"
    + optimal_params_str_rf
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_rf):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_rf.png"
fig, ax = plt.subplots(1, 1)
rf_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_rf, param_dict, fig_path)


# # PLS Regressor - RSCV
# 

# In[104]:


from sklearn import cross_decomposition

pls = cross_decomposition.PLSRegression()
pls_rscv_results_file = f"{RESULTS_PATH}/pls_rscv_results.csv"

rscv(
    features=X,
    target=y,
    model=pls,
    param_distributions=pls_rscv_parameters,
    results_file=pls_rscv_results_file,
    random_state=random_state,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs
)


# In[105]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

# import the optimal model parameters
pls_results = pd.read_csv(pls_rscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_pls = pls_results.loc[pls_results["RMSE"].idxmin(), "params"]
optimal_params_pls = dict(eval(optimal_params_str_pls))
# fit the data with the optimal model parameters
pls_opt = cross_decomposition.PLSRegression(**optimal_params_pls)

y_pred_pls = pls_opt.fit(X_train, y_train).predict(X_test)


# In[106]:


# plot
title_str = (
    f"PLS: Actual vs. Predicted Values \n params:"
    + optimal_params_str_pls
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_pls)):.2f}"
)

param_dict = {"title": title_str}
fig_path = (f"{FIGURE_PATH}/avp_pls.png")

fig, ax = plt.subplots(1, 1)
pls_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_pls, param_dict, fig_path)


# In[29]:


testing


# # KKR Regressor - RSCV

# In[26]:


from sklearn.kernel_ridge import KernelRidge as KRR

krr = KRR()
krr_rscv_results_file = f"{RESULTS_PATH}/krr_rscv_results.csv"



rscv(
    features=X,
    target=y,
    model=krr,
    param_distributions=krr_rscv_parameters,
    results_file=krr_rscv_results_file,
    random_state=random_state,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs
)


# In[108]:


# generate the actual vs. predicted plot

# import the optimal model parameters
krr_results = pd.read_csv(krr_rscv_results_file)

# select the model parameters with the lowest RMSE
optimal_params_str_krr = krr_results.loc[krr_results["RMSE"].idxmin(), "params"]
optimal_params_krr = dict(eval(optimal_params_str_krr))
krr_opt = KRR(**optimal_params_krr)

y_pred_krr = krr_opt.fit(X_train, y_train).predict(X_test)


# In[109]:


# plot
from al_lib.helper_functions import plot_actual_vs_pred

# break the optimal_params_str_krr string into more lines

title_str = (
    f"KRR: Actual vs. Predicted Values \n params:"
    + optimal_params_str_krr
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_krr)):.2f}"
)

param_dict = {"title": title_str}
fig_path_krr = (f"{FIGURE_PATH}/avp_krr.png")

fig, ax = plt.subplots(1, 1)
krr_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_krr, param_dict, fig_path_krr)


# # MLP Regressor - RSCV

# In[110]:


# import mlp
from sklearn.neural_network import MLPRegressor as MLP



mlp = MLP()
mlp_rscv_results_file = f"{RESULTS_PATH}/mlp_rscv_results.csv"

rscv_mpl = rscv(
    features=X,
    target=y,
    model=mlp,
    param_distributions=mlp_rscv_parameters,
    results_file=mlp_rscv_results_file,
    random_state=random_state,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs
)


# In[111]:


# generate the actual vs. predicted plot

# import the optimal model parameters
mlp_results = pd.read_csv(mlp_rscv_results_file)

# select the (optimal) model parameters with the lowest RMSE
optimal_params_str_mlp = mlp_results.loc[mlp_results["RMSE"].idxmin(), "params"]
optimal_params_mlp = dict(eval(optimal_params_str_mlp))

# fit the data with the optimal model parameters
mlp_opt = MLP(**optimal_params_mlp)

y_pred_mlp = mlp_opt.fit(X_train, y_train).predict(X_test)


# In[112]:


# plot
from al_lib.helper_functions import plot_actual_vs_pred

# break the optimal_params_str_mlp string into more lines
optimal_params_str_mlp_break = optimal_params_str_mlp.replace(", ", ",\n")

title_str = (
    f"MLP: Actual vs. Predicted Values \n params:"
    + optimal_params_str_mlp_break
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_mlp)):.2f}"
)

param_dict = {"title": title_str}
fig_path_mlp = (f"{FIGURE_PATH}/avp_mlp.png")

fig, ax = plt.subplots(1, 1)
mlp_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_mlp, param_dict, fig_path_mlp)


# In[26]:


# import xgboost
import xgboost as xgb
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb_rscv_results_file = f"{RESULTS_PATH}/xgb_rscv_results.csv"

rscv_xgb = rscv(
    features=X,
    target=y,
    model=xgb,
    param_distributions=xgb_rscv_parameters,
    results_file=xgb_rscv_results_file,
    random_state=random_state,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs
)


# In[1]:


# generate the actual vs. predicted plot

# import the optimal model parameters
xgb_results = pd.read_csv(xgb_rscv_results_file)

# round the results to 4 decimal places
xgb_results = xgb_results.round(4)

# select the model parameters with the lowest RMSE
optimal_params_str_xgb = xgb_results.loc[xgb_results["RMSE"].idxmin(), "params"]
optimal_params_xgb = dict(eval(optimal_params_str_xgb))

# fit the data with the optimal model parameters
xgb_opt = XGBRegressor(**optimal_params_xgb)

y_pred_xgb = xgb_opt.fit(X_train, y_train).predict(X_test)


# In[115]:


# plot
from al_lib.helper_functions import plot_actual_vs_pred

# break the optimal_params_str_ string into more lines
optimal_params_str_xgb_break = optimal_params_str_xgb.replace(", ", ",\n")

title_str = (
    f"XGB: Actual vs. Predicted Values \n params:"
    + optimal_params_str_xgb_break
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}"
)

param_dict = {"title": title_str}
fig_path = (f"{FIGURE_PATH}/avp_xgb.png")

fig, ax = plt.subplots(1, 1)
xgb_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_xgb, param_dict, fig_path)


# # HGB

# In[30]:


# HGB
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

hbg = HGB()
hgb_rscv_results_file = f"{RESULTS_PATH}/hgb_rscv_results.csv"

rscv_hgb = rscv(
    features=X,
    target=y,
    model=hbg,
    param_distributions=hgb_rscv_parameters,
    results_file=hgb_rscv_results_file,
    random_state=random_state,
    NoTrials=NoTrials,
    nfolds=nfolds,
    n_jobs=n_jobs
)


# In[1]:


# generate the actual vs. predicted plot
hgb_results = pd.read_csv(hgb_rscv_results_file)

# select the model parameters with the lowest RMSE
# select the model parameters with the lowest RMSE
optimal_params_str_hgb = hgb_results.loc[hgb_results["RMSE"].idxmin(), "params"]
optimal_params_hgb = dict(eval(optimal_params_str_hgb))

# fit the data with the optimal model parameters
hgb_opt = HGB(**optimal_params_hgb)

y_pred_hgb = hgb_opt.fit(X_train, y_train).predict(X_test)


# In[ ]:


# plot
from al_lib.helper_functions import plot_actual_vs_pred

# break the optimal_params_str_ string into more lines
optimal_params_str_hgb_break = optimal_params_str_hgb.replace(", ", ",\n")


title_str = (
    f"HGB: Actual vs. Predicted Values \n params:"
    + optimal_params_str_hgb_break
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_hgb):.2f}"
)
fig_path = (f"{FIGURE_PATH}/avp_xgb.png")
param_dict = {"title": title_str}

fig, ax = plt.subplots(1, 1)
xgb_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_xgb, param_dict, fig_path)


# # Quality Control
# 
# In this section the goal is to document the packages which where used during the execution of this notebook

# In[ ]:


## Package informations
from sklearn import show_versions

show_versions()


# In[ ]:


import subprocess
import os
import datetime

# Add data/time information
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")

# Create the output name from the notebookname

output_name = f"{notebook_name.split('.')[0]}_{date}.html"


# Function to convert the notebook to HTML
def convert_notebook_to_html(notebook_name, output_name):
    # Use subprocess to call the jupyter nbconvert command
    subprocess.call(["jupyter", "nbconvert", "--to", "html", notebook_name])
    # Rename the output file
    os.rename(notebook_name.split(".")[0] + ".html", output_name)


# Wait for a short period to ensure all cells have finished executing
time.sleep(5)  # Adjust the sleep duration as needed

# Convert the notebook to HTML
convert_notebook_to_html(notebook_name, output_name)

