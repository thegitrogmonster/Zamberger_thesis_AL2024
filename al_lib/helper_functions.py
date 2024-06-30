import numpy as np
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# This file is used to document functions which are used in the "Thesis" project


def rmse_func(a, b):
    """
    Compute the Root Mean Squared Error (RMSE)

    Input:
    a: array-like, true values of the target variable
    b: array-like, predicted values of the target variable

    Output:
    float, RMSE of the model
    """
    return np.sqrt(mean_squared_error(a, b))


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
    """Function to plot the feature importance of the model if the model
    has the attribute "feature_importances_"

    Parameters
    ----------
        ax: matplotlib axis
            "The Axes class represents one (sub-)plot in a figure" from a matplotlib object
        model: sklearn model
            the fitted model to extract the feature importance
        X_train: pd.DataFrame
            the training set used to fit the model
        param_dict: dict
            additional parameters to pass to the plot, e.g. title: (param_dict["title"])
        fig_path: str (optional)
            path to save the figure. If None, no file is created
    """
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


def import_dpsDeriv1200(datafile):
    """
    Import and perform the necessary preprocessing steps on the dataset "dpsDeriv1200".
    Preprocessing steps:
    - Import the dataset
    - Rename the columns: remove the "X" in the column names

    Parameters
    ----------
    datafile : str
        Path to the dataset "dpsDeriv1200".

    Returns
    -------
    pd.DataFrame
        DataFrame containing the preprocessed dataset "dpsDeriv1200".
    """

    # Import
    data_dps_deriv_1200 = pd.read_csv(datafile, sep=",", decimal=".", encoding="utf-8")
    # rename the columns
    data_dps_deriv_1200 = data_dps_deriv_1200.rename(
        columns=lambda x: x.replace("X", "")
    )
    return data_dps_deriv_1200


def calculate_percentage(size, total_size):
    """
    Calculate the percentage of a subset in relation to the total size.

    Parameters
    ----------
    size : int
        Size of the subset.
    total_size : int
        Total size of the dataset.

    Returns
    -------
    float
        Percentage of the subset in relation to the total size.
    """
    return round((size / total_size) * 100, 2)


def calc_set_sizes(X_train, X_test, X_val, logging):
    """
    Log the number of samples and their percentages of the training, test, and validation sets.

    Parameters
    ----------
    X_train, X_test, X_val : array-like
        Arrays representing the training, test, and validation sets respectively.
    logging : logging.Logger
        Logger object to log the information.

    Returns
    -------
    None
    """
    total_size = len(X_train) + len(X_test) + len(X_val)

    # Log the Number of samples and their percentages of the training, test, and validation set
    logging.info(
        f"Training set: {len(X_train)} ({calculate_percentage(len(X_train), total_size)}%)"
    )
    logging.info(
        f"Test set: {len(X_test)} ({calculate_percentage(len(X_test), total_size)}%)"
    )
    logging.info(
        f"Validation set: {len(X_val)} ({calculate_percentage(len(X_val), total_size)}%)"
    )

    return None


def _validate_parameters(
    X_train,
    y_train,
    model_class=None,
    model_params={},
    selection_criterion=None,
    n_iterations=None,
    n_samples_per_it=None,
    init_sample_size=None,
):
    """
    Validates the parameters passed to the active_learning function.

    Raises
    ------
    ValueError
        If any of the parameters do not meet the validation criteria.
    """
    # Check if X_train and y_train are compatible
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")

    # Check if the length of X_train is greater than the initial sample size or the n_iterations
    if (
        len(X_train) < init_sample_size
        or len(X_train) < n_iterations
        or len(X_train) < (n_samples_per_it**n_iterations)
    ):
        raise ValueError(
            f"X_train does not contain enough samples{len(X_train)} for the initial sample size {len(init_sample_size)} or the number of iterations{n_iterations}. Try reducing the number of iterations or the initial sample size."
        )

    # Check if n_iterations is a positive integer
    if not isinstance(n_iterations, int) or n_iterations <= 0:
        raise ValueError("n_iterations must be a positive integer.")

    # Check if params is a dictionary
    if not isinstance(model_params, dict):
        raise ValueError("model_params must be a dictionary.")

    # Initialize the model without fitting
    try:
        test_model = model_class(**model_params)
    except Exception as e:
        raise ValueError(f"Invalid parameters for {model_class}: {e}")
    return None


def _rnd_initial_sampling(
    X_Pool, X_Learned, y_Pool, y_Learned, init_sample_size, random_state
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Randomly select a subset of samples from the dataset.

    Parameters
    ----------
    X : array-like
        Features of the dataset.
    y : array-like
        Target values of the dataset.
    init_sample_size : int
        Number of samples to select.
    random_state : int
        Seed for the random number generator.

    Returns
    -------
    X_Learned : pd.DataFrame
        Features of the samples selected for training.
    y_Learned : pd.DataFrame
        Target values of the samples selected for training.
    X_Pool : pd.DataFrame
        Features of the samples not selected for training.
    y_Pool : pd.DataFrame
        Target values of the samples not selected for training.
    """

    rng = np.random.default_rng(random_state)
    random_sample_index = rng.choice(X_Pool.index, size=init_sample_size, replace=False)
    x_new = X_Pool.loc[random_sample_index]
    y_new = y_Pool.loc[random_sample_index]
    X_Learned = pd.concat([X_Learned, x_new], ignore_index=True)
    y_Learned = pd.concat([y_Learned, y_new], ignore_index=True)
    X_Pool = X_Pool.drop(index=random_sample_index)
    y_Pool = y_Pool.drop(index=random_sample_index)
    assert all(y_Learned.index == X_Learned.index)
    assert all(y_Pool.index == X_Pool.index)
    return X_Learned, y_Learned, X_Pool, y_Pool

def _create_test_data(n=100, logging = None):
    """
    Create a synthetic dataset for testing purposes. 
    The dataset contains n (default: 100) samples with 3 features and 1 target variable.
    The data is split into training(), test, and validation sets.

    Returns
    -------

    X_train, X_test, X_val, y_train, y_test, y_val
    """
    # Create a synthetic dataset
    data = {
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "feature3": np.random.rand(100),
        "target": np.random.rand(100),
    }
    df = pd.DataFrame(data)
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]
    # split the data into training, test, and validation sets
    random_state = 12345
    validation_size = 0.1
    test_size = 0.3
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
    calc_set_sizes(X_train, X_test, X_val, logging = logging)
    return X_train, X_test, X_val, y_train, y_test, y_val

def _test_params_krr():
    """
    Create a dictionary of parameters for testing the KernelRidge model.

    Returns
    -------
    dict
        Dictionary containing the parameters for testing the KernelRidge model.
    """
    params = {
        "alpha": 1.0,
        "kernel": "linear",
        "gamma": None,
        "degree": 3,
        "coef0": 1,
    }
    return params