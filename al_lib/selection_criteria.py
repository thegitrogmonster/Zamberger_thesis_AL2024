def gsy_selection_(X_Pool, X_Learned, y_Learned, y_pred_pool):
    """Function to select samples in a greedy way in the Y

    Args:
        X_Pool (_type_): Variables of Samples currently available to be selected
        X_Learned (_type_): Variables of Samples already included in the modelling
        y_Learned (_type_): Target values of Samples already included in the modelling
        y_pred_pool (_type_): Target values predicted by the model for the samples in X_Pool

    Returns:
        sample_id (int): Index of the sample in X_Pool with the largest minimum distance to the samples in y_Learned
    """
    # prepare dataframe for the distances
    distances = pd.DataFrame(index=X_Pool.index, columns=X_Learned.index)
    # calculate the distances for each sample in X_Pool to all samples in y_Learned
    for row in distances.index:
        for col in distances.columns:
            distances.loc[row, col] = np.linalg.norm(
                y_pred_pool.loc[row] - y_Learned.loc[col]
            )

    # distances:
    # index: X_Pool.index
    # columns: X_Learned.index
    # calculate the minimum distance for each sample in X_Pool
    distances["min_dist_per_sample"] = distances.min(axis=1)
    # min_dist_per_sample:
    # index: X_Pool.index
    # value: minimum distance to the samples in y_Learned
    # retrieve the sample_id with the largest minimum distance
    sample_id = distances["min_dist_per_sample"].idxmax()
    return sample_id
