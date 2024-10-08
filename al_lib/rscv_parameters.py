import numpy as np
from scipy.stats import randint, uniform, loguniform

# Randomized Search Parameters
rf_rscv_parameters = {
    "n_estimators": randint(low=3, high=100),  # for hyperparameter with discrete values
    "min_samples_split": randint(low=2, high=20),
    "max_features": randint(low=1, high=8),
    "max_depth": randint(low=1, high=20),
}

# Partial Least Squares
pls_rscv_parameters = {
    "n_components": randint(low=1, high=100),
    "scale": [True, False], 
    "max_iter": randint(low=2, high=700),
    "tol": uniform(loc=1e-06, scale=1), # tolerance used as convergence criteria 
    "copy": [True],
}

# KRR-RBF

# previously tested distributions, with no success,
# failure due to the small parameters space being to restrictive
    # alpha_dist = uniform(loc=1e-3, scale=1e-1) # controls 'complexity'
    # gamma_dist = uniform(loc=1e-3, scale=1e-1)
    # kernel_list = ['linear', 'poly', 'polynomial', 'rbf',]

# updated distributions
krr_rscv_parameters = {
    'alpha': loguniform(a=1e-6, b=1e6),  # Regularization strength
    'gamma': loguniform(a=1e-3, b=1e6),  # Kernel coefficient for ‘rbf’
    'kernel': ['rbf', 'linear', 'poly', 'polynomial','chi2'],
}

# MLP
mlp_rscv_parameters = {
    "hidden_layer_sizes": randint(low=50, high=200),  # number of neurons in each layer
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": loguniform(a=1e-6, b=1e6),
    "early_stopping": [True, False],
    "validation_fraction": uniform(loc=0.1, scale=0.1),
    "max_iter": randint(low=200, high=500),
}

# xgboost
xgb_rscv_parameters = {
    "n_estimators": randint(low=3, high=100),  # for hyperparameter with discrete values
    "max_depth": randint(low=1, high=20),
    "learning_rate": uniform(loc=0.01, scale=0.1),
    "subsample": uniform(loc=0.5, scale=0.5),
    "colsample_bytree": uniform(loc=0.5, scale=0.5),
    "gamma": uniform(loc=0, scale=0.5),
    "reg_alpha": uniform(loc=0, scale=0.5),
    "reg_lambda": uniform(loc=0, scale=0.5),
}

# Histogram Gradient Boosting
hgb_rscv_parameters = {
    "loss": ["squared_error"],
    "learning_rate": uniform(loc=0.01, scale=0.1),
    "max_iter": randint(low=100, high=500),
    "max_leaf_nodes": randint(low=15, high=100),
    "min_samples_leaf": randint(low=1, high=40),
}

