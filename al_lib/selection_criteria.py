import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import euclidean_distances


def _random_selection(X_Pool, random_state, *args, **kwargs):
    """Select a random sample from the pool of samples

    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected

    Returns:
        random_sample_index (int): Index of the selected sample in X_Pool
        None: random selection does not return any additional information
    """
    rng = np.random.default_rng(random_state)
    random_sample_index = rng.choice(X_Pool.index, replace=False)
    return random_sample_index, None


def _gsx_selection(X_Pool, X_Learned, *args, **kwargs):
    """Function to select samples in a greedy way in X-Space
    select the sample from X_Pool, where the euclidean distance to the samples in X_Learned is the largest
    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected
        X_Learned (pd.DataFrame): Variables of Samples already included in the modelling

    Returns:
        sample_id (int): Index of the sample in X_Pool with the largest distance to the samples in X_Learned
        gsx_for_id (float): Distance of the sample with the largest distance to the samples in X_Learned
    """
    # Greedy Sampling by Euclidean Distance
    distances = euclidean_distances(
        X_Pool, X_Learned
    )  # distances : ndarray of shape (n_samples_X, n_samples_Y)
    distances_df = pd.DataFrame(distances, index=X_Pool.index, columns=X_Learned.index)
    # select the sample with the largest distance
    sample_id = distances_df.sum(axis=1).idxmax()
    # identify the distance for the selected sample
    gsx_for_id = distances_df.sum(axis=1).max()
    return sample_id, gsx_for_id


def _gsy_selection(X_Pool, X_Learned, y_Learned, y_pred_pool,*args, **kwargs):
    """Function to select samples in a greedy way in Y-Space

    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected
        X_Learned (pd.DataFrame): Variables of Samples already included in the modelling
        y_Learned (pd.DataFrame): Target values of Samples already included in the modelling
        y_pred_pool (pd.DataFrame): Predictions of the model for the samples in X_Pool

    Returns:
        sample_id (int): Index of the sample in X_Pool with the largest minimum distance to the samples in y_Learned
        gsy_for_id (float): Minimum distance of the sample with the largest minimum distance to the samples in y_Learned
    """
    # prepare dataframe for the distances
    distances = pd.DataFrame(index=X_Pool.index, columns=X_Learned.index)
    # calculate the distances for each sample in X_Pool to all samples in y_Learned
    for row in distances.index:  # iterate over the samples in X_Pool
        for col in distances.columns:  # iterate over the samples in y_Learned
            # calculate the euclidean distance between the predicted value
            # of each sample in X_Pool and the annotated value for each sample in y_Learned
            distances.loc[row, col] = np.linalg.norm(
                y_pred_pool.loc[row] - y_Learned.loc[col]
            )
    # calculate the minimum distance for each sample in X_Pool
    distances["min_dist_per_sample"] = distances.min(axis=1)
    # retrieve the sample_id with the largest minimum distance
    sample_id = distances["min_dist_per_sample"].idxmax()
    gsy_for_id = distances["min_dist_per_sample"].max()
    return sample_id, gsy_for_id


def _uncertainty_selection(
    X_Pool, y_Pool, X_Learned, y_Learned, random_state, model, n_fold, logging, n_jobs, *args, **kwargs
) -> (int, float):
    """Function to select samples with the highest uncertainty for predictions
    evaluated by the standard deviation of the predictions in a cross-validation
    setting
    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected
        y_Pool (pd.DataFrame): Target values of Samples currently available to be selected
        X_Learned (pd.DataFrame): Variables of Samples already included in the modelling
        y_Learned (pd.DataFrame): Target values of Samples already included in the modelling
        random_state (int): Seed for the random number generator
        model (object): Model object, fitted on the training set previously
        n_fold (int): Number of folds for the cross-validation
        logging (object): Logging object
        n_jobs (int): Number of cores used for processing
    Returns:
        sample_id (int): Index of the sample in X_Pool with the highest uncertainty
        uncertainty_for_id (float): Uncertainty of the sample with the highest uncertainty
    """
    # prepare the predictions df with dim(X_Pool, nfolds)
    predictions = None
    predictions = pd.DataFrame(index=X_Pool.index, columns=range(n_fold))

    # Generate pseudo-targets for X_Pool using the provided model
    y_Pool_pred = model.predict(X_Pool)

    # generate the n-fold splits for X_Pool and y_Pool
    ss = ShuffleSplit(n_splits=n_fold, test_size=0.7, random_state=random_state)
    ss.get_n_splits(X_Pool, y_Pool)

    for n_fold, (train_index, test_index) in enumerate(ss.split(X_Pool, y_Pool)):
        # logging.info(f"size of individual split: {len(train_index), len(test_index)}")
        # merge the X_Learned and the current split of X_Pool into a new training set
        X_train_fold = pd.concat([X_Learned, X_Pool.iloc[train_index]])
        y_train_fold = pd.concat([y_Learned, pd.Series(y_Pool_pred[train_index], index=train_index)])

        # fit the model on the new training set
        model.fit(X_train_fold, y_train_fold)

        # predict the test set and save the predictions in the predictions df
        y_pred = model.predict(X_Pool.iloc[test_index])

        # get the index of the predictions
        y_pred_index = X_Pool.index[test_index]
        # merge the predictions with the predictions df
        predictions.loc[y_pred_index, n_fold] = y_pred

    # calculate the std of the predictions
    predictions["std"] = predictions.std(axis=1)

    # identify the sample with the highest std
    sample_id = predictions["std"].idxmax()
    uncertainty_for_id = predictions["std"].max()
    return sample_id, uncertainty_for_id

def _distance_weighing(
    X_Pool, y_Pool, X_Learned, y_Learned, random_state, model, n_fold, logging, n_jobs, *args, **kwargs
) -> (int, float):
    """IDEAL Distance weighing selection strategy,
    see Bemporad et al. 2019

    Ideal distance weighing selection strategy to select samples weighted by 
    the distance to the samples in the training set. The aquisition function
    integrates the uncertainty for each sample to promote exploration, and the 
    squared euclidean distance to the samples in the training set to promote exploitation.

    alternative weight decay function:
    wk(x) = exp(-d^2(x, X_learned)^2) / (d^2(x, X_learned))	
    
    Normalized weights:
    vk(x)
    vk(x) = wk(x)/sum(wk(x))

    IDW variance function: 
    s2(x) = sum(vk(x) * (y_learned - y_pred(x))^2)

    IDW distance function, euclidean distance, for pure exploration:
    z(x) = d(x, X_learned)

    Acquisiton function:
    aq(x) = (1 + omega * rho(x)) * sum((c(x) * (s2(x) + delta * z(x)))
        where 
            omega: trade off between exploration and exploitation
                default  = 0.5, balanced exploration and exploitation
            rho(x): uncertainty of the sample x (IDW variance of the prediction)
            sum(): sum over all samples in the pool
            c(x): weight function between 0 and 1, uniform: c(x) = 1
            delta: trade off between exploration and exploitation,
                 default  = 0.0, aquisition purely based on IDW variance

            
    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected
        y_Pool (pd.DataFrame): Target values of Samples currently available to be selected
        X_Learned (pd.DataFrame): Variables of Samples already included in the modelling
        y_Learned (pd.DataFrame): Target values of Samples already included in the modelling
        y_pred_pool (pd.DataFrame): Predictions of the model for the samples in X_Pool
        random_state (int): Seed for the random number generator
        model (object): Model object, fitted on the training set previously
        n_fold (int): Number of folds for the cross-validation
        logging (object): Logging object
        n_jobs (int): Number of cores used for processing

    Returns:
        sample_id (int): Index of the sample in X_Pool with the highest uncertainty
        uncertainty_for_id (float): Uncertainty of the sample with the highest uncertainty
    """
    # Prepare a df for the IDW acquisition function one row per sample in X_Pool
    df_idw = pd.DataFrame(index=X_Pool.index)
    # generate a column for each variable in the aquisition function
    df_idw["rho"] = 0
    df_idw["c"] = 1
    df_idw["s2"] = 0
    df_idw["z"] = 0

    distances = euclidean_distances(
        X_Pool, X_Learned
    )  # distances : ndarray of shape (n_samples_X, n_samples_Y)
    distances_df = pd.DataFrame(distances, index=X_Pool.index, columns=X_Learned.index)

    # calculate the std of the predictions for each sample in X_Pool
    # prepare the predictions df with dim(X_Pool, nfolds)
    predictions = None
    predictions = pd.DataFrame(index=X_Pool.index, columns=range(n_fold))

    # Generate pseudo-targets for X_Pool using the provided model
    y_Pool_pred = model.predict(X_Pool)

    # generate the n-fold splits for X_Pool and y_Pool
    ss = ShuffleSplit(n_splits=n_fold, test_size=0.7, random_state=random_state)
    ss.get_n_splits(X_Pool, y_Pool)

    for n_fold, (train_index, test_index) in enumerate(ss.split(X_Pool, y_Pool)):

        # merge the X_Learned and the current split of X_Pool into a new training set
        X_train_fold = pd.concat([X_Learned, X_Pool.iloc[train_index]])
        y_train_fold = pd.concat([y_Learned, pd.Series(y_Pool_pred[train_index], index=train_index)])

        # fit the model on the new training set
        model.fit(X_train_fold, y_train_fold)

        # predict the test set and save the predictions in the predictions df
        y_pred_new = model.predict(X_Pool.iloc[test_index])

        # get the index of the predictions
        y_pred_index = X_Pool.index[test_index]
        # merge the predictions with the predictions df
        predictions.loc[y_pred_index, n_fold] = y_pred_new

    # calculate the std of the predictions
    predictions["std"] = predictions.std(axis=1)

    # wk(x)
    wk = np.exp(-distances_df**2) / distances_df

    # vk(x)
    vk = wk / wk.sum(axis=1)

    # s2(x)
    y_pred_Learned = model.predict(X_Learned)
    s2 = (vk * (y_Learned - y_pred_Learned)**2).sum(axis=1)

    # z(x)
    z = distances_df.sum(axis=1)

    # aq(x)
    if "omega" in kwargs:
        omega = kwargs["omega"]
    else:
        omega = 0.5 #balanced exploration and exploitation
    
    if "delta" in kwargs:
        delta = kwargs["delta"]
    else:
        delta = 0.0 # purely based on IDW variance

    aq = (1 + omega * predictions["std"]) * (vk * (s2 + 0 * z)).sum(axis=1)

    # identify the sample with the highest aq
    sample_id = aq.idxmax()
    idw_for_id = aq.max()

    return sample_id, idw_for_id
