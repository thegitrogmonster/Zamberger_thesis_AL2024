import numpy as np
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import matplotlib.pyplot as plt
# This file is used to document functions which are used in the "Thesis" project


def rmse_func(a, b):
    """
    Compute the Root Mean Squared Error (RMSE)

    Input:
    y_true: array-like, true values of the target variable
    y_pred: array-like, predicted values of the target variable

    Output:
    float, RMSE of the model
    """
    return np.sqrt(mean_squared_error(a, b))

    """
    Report the best hyperparameters and the best score of the model

    Input:
    cv_obj: can be a GridSearchCV or RandomizedSearchCV object from sklearn

    Output:

    """
def report_model(cv_obj):
    """A function to report the best hyperparameters and the best score of the model
    Parameters
    ----------
    cv_obj: can be a GridSearchCV or RandomizedSearchCV object from sklearn   
    Returns
    -------
    None
    """
    model_name = cv_obj.estimator.__class__.__name__
    # print the best hyperparameters and the best score
    print(f"Best hyperparameters for {model_name}: {cv_obj.best_params_}")
    print(f"Best score {cv_obj.best_score_} for {model_name}: {cv_obj.best_score_}")
    # print the best estimator
    print(f"Best estimator for {model_name}: {cv_obj.best_estimator_}")

def plot_actual_vs_pred(ax, y_true, y_pred, param_dict, fig_path=None):
    """
    Plot the actual values against the predicted values. 
    The function tries to emulate the template from the documentation: 
    https://matplotlib.org/3.5.0/tutorials/introductory/usage.html#the-object-oriented-interface-and-the-pyplot-interface

    Parameters
    ----------
    ax: matplotlib.axes.Axes, 
        the axes object to draw the plot onto
        
    y_true: array-like
        true values of the target variable

    y_pred: array-like, 
        predicted values of the target variable

    param_dict: dict,
        dictionary of parameters to pass to the plot function
    fig_path: str, optional(default is None)
        path to save the figure. If None, no file is created
    Returns
    -------
        Plot of the "actual values" against the predicted values.
        Adds a 45° line to estimate the precision of individual predictions.
        return the plot object for further manipulation/investigation
    """
    ax.scatter(y_true, y_pred, alpha=0.5)
    # add a 45 degree line
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=4)
    # label the axis
    ax.set_xlabel("Actual values ['year']")
    ax.set_ylabel("Predicted values ['year']")
    if "title" in param_dict:
        ax.set_title(param_dict["title"])
    else:
        ax.set_title("Actual vs Predicted values")
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        pass
    plt.show()

def plot_feature_importance(ax, model, X_train, param_dict, fig_path):
    try:
        feature_importance = model.feature_importances_
    except AttributeError:
        feature_importance = model.best_estimator_.feature_importances_
    # get the 10 most important features
    indices = np.argsort(feature_importance)[::-1]
    # plot the feature importance for the top 10 features
    ax.barh(range(10), feature_importance[:10], align="center")
    ax.set_yticks(range(10))
    ax.set_yticklabels(X_train.columns)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    if "title" in param_dict:
        ax.set_title(param_dict["title"])
    else:
        ax.set_title(f"Feature importance of the {model.__class__.__name__}")
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        pass
    plt.show()