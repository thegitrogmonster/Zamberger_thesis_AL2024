#!/usr/bin/env python

"""
This Script contains the functions used to create the models for the thesis
"""
# Imports

import numpy as np
import xgboost


def model_rf():
    """
    This function creates the Random Forest model
    """
    from sklearn.ensemble import RandomForestRegressor
    model_rf = RandomForestRegressor()
    return model_rf

def model_xgb():
    """
    This function creates the XGBoost model
    """
    from xgboost import XGBRegressor
    model_xgb = XGBRegressor()
    return model_xgb

def model_pls():
    """
    This function creates the PLS model
    """
    from sklearn.cross_decomposition import PLSRegression
    model_pls = PLSRegression()
    return model_pls

def model_rbf():
    """
    This function creates the RBF model
    """
    from sklearn.kernel_ridge import KernelRidge
    model_rbf = KernelRidge(kernel='rbf')
    return model_rbf

def model_mlp():
    """
    This function creates the MLP model
    """
    from sklearn.neural_network import MLPRegressor
    model_mlp = MLPRegressor()
    return model_mlp