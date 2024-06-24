import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import euclidean_distances


def _random_selection(X_Pool, *args, **kwargs):
    """Select a random sample from the pool of samples

    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected

    Returns:
        random_sample_index (int): Index of the selected sample in X_Pool
        None: random selection does not return any additional information
    """
    random_sample_index = np.random.choice(X_Pool.index)
    return random_sample_index, None


def _gsx_selection(X_Pool, X_Learned):
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


def _gsy_selection(X_Pool, X_Learned, y_Learned, y_pred_pool):
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
    X_Pool, y_Pool, X_Learned, y_Learned, y_pred_pool, model, n_fold, logging
) -> (int, float):
    """Function to select samples with the highest uncertainty for predictions
    evaluated by the standard deviation of the predictions in a cross-validation
    setting
    Parameters:
        X_Pool (pd.DataFrame): Variables of Samples currently available to be selected
        y_Pool (pd.DataFrame): Target values of Samples currently available to be selected
        X_Learned (pd.DataFrame): Variables of Samples already included in the modelling
        y_Learned (pd.DataFrame): Target values of Samples already included in the modelling
        y_pred_pool (pd.DataFrame): Predictions of the model for the samples in X_Pool
        model (object): Model object, fitted on the training set previously
        n_fold (int): Number of folds for the cross-validation
        logging (object): Logging object
    Returns:
        sample_id (int): Index of the sample in X_Pool with the highest uncertainty
        uncertainty_for_id (float): Uncertainty of the sample with the highest uncertainty
    """
    # prepare the predictions df with dim(X_Pool, nfolds)
    predictions = None
    predictions = pd.DataFrame(index=X_Pool.index, columns=range(n_fold))
    # pred_uncertainty.index = X_Pool.index
    logging.info(
        f"Shape X_Pool:{X_Pool.shape}" + f"Shape y_Pool{y_Pool.shape} before sss"
    )

    # generate the n-fold splits for X_Pool and y_Pool
    ss = ShuffleSplit(n_splits=n_fold, test_size=0.7, random_state=42)
    ss.get_n_splits(X_Pool, y_Pool)

    for n_fold, (train_index, test_index) in enumerate(ss.split(X_Pool, y_Pool)):
        logging.info(f"size of individual split: {len(train_index), len(test_index)}")
        # merge the X_Learned and the current split of X_Pool into a new training set
        X_train_fold = pd.concat([X_Learned, X_Pool.iloc[train_index]])
        y_train_fold = pd.concat([y_Learned, y_Pool.iloc[train_index]])
        # logging.info(f"Shapes: n_Fold: {n_fold} X_train_fold :{X_train_fold.shape}"+f"Shape y_train_fold{y_train_fold.shape} during sss")
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
