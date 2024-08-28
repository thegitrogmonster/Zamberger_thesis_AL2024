#!/usr/bin/env python
# coding: utf-8

# The code written herein aims to generate clean plots to include in the thesis. 
# 
# The plots created during the modelling and AL processes contain technical intresting information, but are not always suitable for the thesis. 
# Therefor selected plots are recreated here. 

# # Setup

# In[112]:


import os

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"


# In[113]:


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


# In[114]:


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


# # Import Data

# In[115]:


data_dps_deriv_1200 = pd.read_csv(
    DATA_PATH + "/dpsDeriv1200.csv", sep=",", decimal=".", encoding="utf-8"
)
data_dps_deriv_1200 = data_dps_deriv_1200.rename(columns=lambda x: x.replace("X", ""))
data_dps_deriv_1200 = data_dps_deriv_1200.rename(columns={"Unnamed: 0": "Samplename"})
data_dps_deriv_1200


# In[116]:


# Switch for the dataset
# Select from (data_small, data_full, data_2nd_deriv) or other if implemented
data_raw = data_dps_deriv_1200
data_raw.dataset_name = "data_dps_deriv_1200"


# In[117]:


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

else:
    nfolds = 10
    NoTrials = 15
    n_jobs = 30
    save_model = True
    data = data_raw


# In[118]:


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


# In[119]:


# Split into target and features
# The goal is to predict the year column of the dataset using the spectral data
X = data.select_dtypes("float")
y = data["year"]
X.shape, y.shape


# In[120]:


# Split the data into training and test sets
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)



# In[121]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape
# assert the shapes and raise an error if they are not equal
assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
assert y_train.shape[0] + y_test.shape[0] == y.shape[0]


# In[122]:


from sklearn.metrics import make_scorer
from sklearn.metrics import root_mean_squared_error

# create a scorer which calculates Root Mean Squeared Error (RMSE)

scoring = make_scorer(root_mean_squared_error, greater_is_better=False)


# # RF

# In[123]:


# Actual-vs-Predicted plot for the best model

# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred

rf = RandomForestRegressor()
rf_rscv_results_file = f"{RESULTS_PATH}rf_rscv_results.csv"

# import the optimal model parameters
rf_results = pd.read_csv(rf_rscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_rf = rf_results.loc[rf_results["RMSE"].idxmin(), "params"]
optimal_params_rf = dict(eval(optimal_params_str_rf))

rf_opt = RandomForestRegressor(**optimal_params_rf)

y_pred_rf = rf_opt.fit(X_train, y_train).predict(X_test)

title_str = (
    f"Actual vs. Predicted Values (Random Forests)"
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_rf):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_rf_rscv.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
rf_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_rf, param_dict, fig_path)


# # PLS

# In[124]:


from sklearn import cross_decomposition

pls = cross_decomposition.PLSRegression()
pls_rscv_results_file = f"{RESULTS_PATH}/pls_rscv_results.csv"

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

# plot
title_str = (
    f"Actual vs. Predicted Values (PLS)"
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_pls)):.2f}"
)

param_dict = {"title": title_str}
fig_path = (f"{FIGURE_PATH}/avp_pls_rscv.png")

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
pls_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_pls, param_dict, fig_path)


# # KRR

# In[125]:


from sklearn.kernel_ridge import KernelRidge as KRR

krr = KRR()
krr_rscv_results_file = f"{RESULTS_PATH}/krr_rscv_results.csv"


# generate the actual vs. predicted plot

# import the optimal model parameters
krr_results = pd.read_csv(krr_rscv_results_file)

# select the model parameters with the lowest RMSE
optimal_params_str_krr = krr_results.loc[krr_results["RMSE"].idxmin(), "params"]
optimal_params_krr = dict(eval(optimal_params_str_krr))
krr_opt = KRR(**optimal_params_krr)

y_pred_krr = krr_opt.fit(X_train, y_train).predict(X_test)

# plot

# break the optimal_params_str_krr string into more lines

title_str = (
    f"Actual vs. Predicted Values (KRR)"
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_krr)):.2f}"
)

param_dict = {"title": title_str}
fig_path_krr = (f"{FIGURE_PATH}/avp_krr_rscv.png")

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
krr_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_krr, param_dict, fig_path_krr)


# # MLP

# In[126]:


# import mlp
from sklearn.neural_network import MLPRegressor as MLP

mlp = MLP()
mlp_rscv_results_file = f"{RESULTS_PATH}/mlp_rscv_results.csv"

# generate the actual vs. predicted plot

# import the optimal model parameters
mlp_results = pd.read_csv(mlp_rscv_results_file)

# select the (optimal) model parameters with the lowest RMSE
optimal_params_str_mlp = mlp_results.loc[mlp_results["RMSE"].idxmin(), "params"]
optimal_params_mlp = dict(eval(optimal_params_str_mlp))

# fit the data with the optimal model parameters
mlp_opt = MLP(**optimal_params_mlp)

y_pred_mlp = mlp_opt.fit(X_train, y_train).predict(X_test)

# plot

# break the optimal_params_str_mlp string into more lines
optimal_params_str_mlp_break = optimal_params_str_mlp.replace(", ", ",\n")

title_str = (
    f"Actual vs. Predicted Values (MLP)"
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_mlp)):.2f}"
)

param_dict = {"title": title_str}
fig_path_mlp = (f"{FIGURE_PATH}/avp_mlp_rscv.png")

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
mlp_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_mlp, param_dict, fig_path_mlp)


# # XGB

# In[127]:


# import xgboost
import xgboost as xgb
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb_rscv_results_file = f"{RESULTS_PATH}/xgb_rscv_results.csv"

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

# plot
from al_lib.helper_functions import plot_actual_vs_pred

# break the optimal_params_str_ string into more lines
optimal_params_str_xgb_break = optimal_params_str_xgb.replace(", ", ",\n")

title_str = (
    f"Actual vs. Predicted Values(XGBoost)"
    + f"\n RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}"
)

param_dict = {"title": title_str}
fig_path = (f"{FIGURE_PATH}/avp_xgb_rscv.png")

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
xgb_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_xgb, param_dict, fig_path)


# # HGB

# In[128]:


# HGB
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

hbg = HGB()
hgb_rscv_results_file = f"{RESULTS_PATH}/hgb_rscv_results.csv"

# generate the actual vs. predicted plot
hgb_results = pd.read_csv(hgb_rscv_results_file)

# select the model parameters with the lowest RMSE
# select the model parameters with the lowest RMSE
optimal_params_str_hgb = hgb_results.loc[hgb_results["RMSE"].idxmin(), "params"]
optimal_params_hgb = dict(eval(optimal_params_str_hgb))

# fit the data with the optimal model parameters
hgb_opt = HGB(**optimal_params_hgb)

y_pred_hgb = hgb_opt.fit(X_train, y_train).predict(X_test)

# plot

# break the optimal_params_str_ string into more lines
optimal_params_str_hgb_break = optimal_params_str_hgb.replace(", ", ",\n")


title_str = (
    f"Actual vs. Predicted Values (HGB)"
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_hgb):.2f}"
)
fig_path = (f"{FIGURE_PATH}/avp_hgb_rscv.png")
param_dict = {"title": title_str}

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
xgb_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_xgb, param_dict, fig_path)


# # GSCV - results
# 
# ## Setup

# In[129]:


# Results path
RESULTS_PATH = basepath + "03_Modelling/03_2_gscv/gscv_results/"

# Figure path
FIGURE_PATH = basepath + "03_Modelling/03_2_gscv/gscv_figures/"

from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.neural_network import MLPRegressor as MLP
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.ensemble import RandomForestRegressor


# ## RF

# In[130]:


# generate the actual vs. predicted plot
from al_lib.helper_functions import plot_actual_vs_pred
rf_gscv_results_file = f"{RESULTS_PATH}rf_gscv_results.csv"
# import the optimal model parameters
rf_results = pd.read_csv(rf_gscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_rf = rf_results.loc[rf_results["RMSE"].idxmin(), "params"]
optimal_params_rf = dict(eval(optimal_params_str_rf))

rf_opt = RandomForestRegressor(**optimal_params_rf, random_state=random_state)

y_pred_rf = rf_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_rf.replace("np.int64", "").replace("(", "").replace(")", "")

title_str = (
    f"Actual vs. Predicted Values (Random Forests)"
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_rf):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_rf_gscv.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
rf_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_rf, param_dict, fig_path)


# ## PLS

# In[131]:


pls_gscv_results_file = f"{RESULTS_PATH}pls_gscv_results.csv"

# import the optimal model parameters
pls_results = pd.read_csv(pls_gscv_results_file)
# select the model parameters with the lowest RMSE
optimal_params_str_pls = pls_results.loc[pls_results["RMSE"].idxmin(), "params"]
optimal_params_pls = dict(eval(optimal_params_str_pls))

pls_opt = PLSRegression(**optimal_params_pls)

y_pred_pls = pls_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_pls.replace("np.int64", "").replace("(", "").replace(")", "")

title_str = (
    f"PLS Regression: Actual vs. Predicted Values"
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_pls):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_pls_gscv.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
rf_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_pls, param_dict, fig_path)


# ## KRR

# In[132]:


krr = KRR()

# Define the results file
krr_gscv_results_file = f"{RESULTS_PATH}krr_gscv_results.csv"

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
    f"Kernel-Ridge Regression: Actual vs. Predicted Values"
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_krr):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_krr_gscv.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
krr_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_krr, param_dict, fig_path)


# ## MLP

# In[133]:


# import mlp
from sklearn.neural_network import MLPRegressor as MLP
mlp = MLP()

mlp_gscv_results_file = f"{RESULTS_PATH}/mlp_gscv_results.csv"

# import parameters
mlp_results = pd.read_csv(mlp_gscv_results_file)
optimal_params_str_mlp = mlp_results.loc[mlp_results["RMSE"].idxmin(), "params"]
optimal_params_mlp = dict(eval(optimal_params_str_mlp))

mlp_opt = MLP(**optimal_params_mlp)

y_pred_mlp = mlp_opt.fit(X_train, y_train).predict(X_test)

title_str_params = optimal_params_str_mlp.replace("np.int64", "").replace("(", "").replace(")", "").replace("np.float64", "")

title_str = (
    f"Actual vs. Predicted Values (MLP)"
    + f"\n RMSE = {root_mean_squared_error(y_test, y_pred_mlp):.2f}"
)

param_dict = {"title": title_str}
fig_path = f"{FIGURE_PATH}avp_mlp_gscv.png"
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
mlp_avp_plot = plot_actual_vs_pred(ax, y_test, y_pred_mlp, param_dict, fig_path)

