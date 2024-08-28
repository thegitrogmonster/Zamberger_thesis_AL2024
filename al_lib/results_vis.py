import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def load_data(filename, TABLES_PATH):
    data = pd.read_csv(TABLES_PATH + filename)
    return data

def _seperate_results_test(results, model = None, model_name = None): 

    test_rsme_random = pd.DataFrame()
    test_rsme_gsx = pd.DataFrame()
    test_rsme_gsy = pd.DataFrame()
    test_rmse_uncertainty = pd.DataFrame()
    test_rmse_idw = pd.DataFrame()
    if model_name == None:
        model_name = str(model).split(" ")[1]
    if model == None:
        model_name = model_name
    if (model_name is None) & (model is None):
        raise ValueError("Please provide the model or model_name")
    test_rsme_random = pd.concat(
        [
            test_rsme_random,
            results.filter(regex=f"rmse_test_{model_name}_Random"),
        ],
        axis=1,
    )
    test_rsme_gsx = pd.concat(
        [
            test_rsme_gsx,
            results.filter(
                regex=f"rmse_test_{model_name}_GSx"
            ),
        ],
        axis=1,
    )
    test_rsme_gsy = pd.concat(
        [
            test_rsme_gsy,
            results.filter(
                regex=f"rmse_test_{model_name}_GSy"
            ),
        ],
        axis=1,
    )
    test_rmse_uncertainty = pd.concat(
        [
            test_rmse_uncertainty,
            results.filter(
                regex=f"rmse_test_{model_name}_Variance"
            ),
        ],
        axis=1,
    )
    test_rmse_idw = pd.concat(
        [
            test_rmse_idw,
            results.filter(
                regex=f"rmse_test_{model_name}_IDW"
            ),
        ],
        axis=1,
    )
    
    return test_rsme_random, test_rsme_gsx, test_rsme_gsy, test_rmse_uncertainty, test_rmse_idw

def _seperate_results_val(results, model = None, model_name = None): 

    val_rsme_random = pd.DataFrame()
    val_rsme_gsx = pd.DataFrame()
    val_rsme_gsy = pd.DataFrame()
    val_rmse_uncertainty = pd.DataFrame()
    val_rmse_idw = pd.DataFrame()
    if model_name == None:
        model_name = str(model).split(" ")[1]
    if model == None:
        model_name = model_name
    if (model_name is None) & (model is None):
        raise ValueError("Please provide the either model or model_name")
    val_rsme_random = pd.concat(
        [
            val_rsme_random,
            results.filter(regex=f"rmse_val_{model_name}_Random"),
        ],
        axis=1,
    )
    val_rsme_gsx = pd.concat(
        [
            val_rsme_gsx,
            results.filter(
                regex=f"rmse_val_{model_name}_GSx"
            ),
        ],
        axis=1,
    )
    val_rsme_gsy = pd.concat(
        [
            val_rsme_gsy,
            results.filter(
                regex=f"rmse_val_{model_name}_GSy"
            ),
        ],
        axis=1,
    )
    val_rmse_uncertainty = pd.concat(
        [
            val_rmse_uncertainty,
            results.filter(
                regex=f"rmse_val_{model_name}_Variance"
            ),
        ],
        axis=1,
    )
    val_rmse_idw = pd.concat(
        [
            val_rmse_idw,
            results.filter(
                regex=f"rmse_val_{model_name}_IDW"
            ),
        ],
        axis=1,
    )
    return val_rsme_random, val_rsme_gsx, val_rsme_gsy, val_rmse_uncertainty, val_rmse_idw

def _plot_rmse(test_rmse, selection_criteria, model_name = None, model = None, title = None, filepath = None):
    if model_name == None:
        model_name = str(model).split(" ")[1]
    if model == None:
        model_name = model_name
    if (model_name is None) & (model is None):
        raise ValueError("Please provide the model or model_name")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for i, rmse in enumerate(test_rmse):
        ax.plot(rmse.mean(axis=1), label=selection_criteria[i]['crit_name'])
        # calculate the standard deviation
        std = rmse.std(axis=1)
        ax.fill_between(
            range(len(rmse)),
            rmse.mean(axis=1) - std,
            rmse.mean(axis=1) + std,
            alpha=0.2,
        )
    if title == None:
        title = f"Test RMSE for {model_name}"
    else:
        title = title
    ax.set_title(title)
    ax.set_xlabel("Number of iterations of Active Learning")
    ax.set_ylabel("RMSE")
    ax.legend()
    if filepath != None:
        plt.savefig(filepath)
    plt.show()
    return fig, ax

def combined_auc_plot(mean_auc_test, mean_auc_val, model_name, filepath=None, selection_criteria = None):
    """Generate a barplot of the mean AUC for the test and validation set for each selection criteria
    ----------
    Parameters:
    mean_auc_test: list of tuples, the mean AUC for the test set for each selection criteria
    mean_auc_val: list of tuples, the mean AUC for the validation set for each selection criteria
    model_name: str, the name of the model
    filepath: str, the path to save the plot
    ----------
    Returns:
    fig, ax: the figure and axis of the plot"""
    # select two nice colors
    colour_selection = ["#9ecae1", "#3182bd"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    index = np.arange(len(selection_criteria))
    barwidth = 0.35
    # plot the test aucs
    # correct code
    for i, (_, _, auc_mean, auc_err) in enumerate(mean_auc_test):
        ax.bar(
            index[i] - barwidth / 2,
            auc_mean,
            yerr=auc_err,
            label=f"{selection_criteria[i]['crit_name']} test",
            align="center",
            width=barwidth,
            color=colour_selection[0],
            capsize=6,
        )
    for i, (_, _, auc_mean, auc_err) in enumerate(mean_auc_val):
        ax.bar(
            index[i] + barwidth / 2,
            auc_mean,
            yerr=auc_err,
            label=f"{selection_criteria[i]['crit_name']} val",
            align="center",
            width=barwidth,
            color=colour_selection[1],
            capsize=6,
        )
    title = f"Mean RMSE of Active Learning Process for {model_name} \nComparison of Selection Strategy for Test and Validation Set"

    # incorrect code
    # for i, auc in enumerate(aucs_test):
    #     ax.bar(selection_criteria[i]['crit_name'], auc[0], yerr=auc[1], label=f"{selection_criteria[i]['crit_name']} test", align='center', width=barwidth )
    # for i, auc in enumerate(auc_val):
    #     ax.bar(selection_criteria[i]['crit_name'], auc[0], yerr=auc[1], label=f"{selection_criteria[i]['crit_name']} val", align='center',width=barwidth)
    # title = f"AUC for {model_name}"
    ax.set_title(title)
    ax.set_ylabel("mean RMSE +/- std")
    fig.legend = ax.legend()
    # x_labels = [
    #     selection_criteria[i]["crit_name"] for i in range(len(selection_criteria))
    # ]
    x_labels = ["Random", "GSx", "GSy", "Variance", "IDW-variant"]
    fig.xticks = ax.set_xticks(
        [i + 0.001 * barwidth for i in index], x_labels, # rotation=45
    )
    # fix the legend
    c0_patch = mpatches.Patch(color=colour_selection[0], label="Test Set")
    c1_patch = mpatches.Patch(color=colour_selection[1], label="Validation SET")

    ax.legend(
        handles=[c0_patch, c1_patch],
    )
    if filepath != None:
        plt.savefig(filepath + f"combined_auc_{model_name}.png")
    plt.show()
    return fig, ax


def test_combined_auc_plot():
    mean_auc_test = _calculate_auc(test_rmse)
    mean_auc_val = _calculate_auc(sep_res_val)
    combined_auc_plot(mean_auc_test, mean_auc_val, model_name="KRR")


def _seperate_results_test_batch(results, model = None, model_name = None, batch_sizes = None): 
    """Seperates the results dataframe into the test rmse for each selection criteria

    Args:
        results (pandas dataframe): results dataframe, create during Active Learning in batch mode
        model (model object): either model or model name needs to be provided, 
            to seperate the results accordingly, model name is preferred. 
            Defaults to None
        model_name (str, optional): model name as string matching the column 
        titles. Necessary to seperate the data Defaults to None.
        batch_sizes (list, optional): list of batch sizes to seperate the results

    Raises:
        ValueError: If neither model name nor model is provided, there is no 
        basis to identify the corresponding model string in the results
         dataframe/column names

    Returns: 
        test_rsme_random, test_rsme_gsx, test_rsme_gsy, test_rmse_uncertainty, test_rmse_idw
        Description: every object represents the test rmse for the corresponding selection criteria
    """

    test_rsme_random = pd.DataFrame()
    test_rsme_gsx = pd.DataFrame()
    test_rsme_gsy = pd.DataFrame()
    test_rmse_uncertainty = pd.DataFrame()
    test_rmse_idw = pd.DataFrame()
    if model_name == None:
        model_name = str(model).split(" ")[1]
    if model == None:
        model_name = model_name
    if (model_name is None) & (model is None):
        raise ValueError("Please provide the model or model_name")
    if batch_sizes == None:
        raise ValueError("Please provide the batch sizes")
    for batch_size in batch_sizes:
        test_rsme_random = pd.concat(
            [
                test_rsme_random,
                results.filter(regex=f"rmse_test_{batch_size}_{model_name}_random"),
            ],
            axis=1,
        )
        test_rsme_gsx = pd.concat(
            [
                test_rsme_gsx,
                results.filter(
                    regex=f"rmse_test_{batch_size}_{model_name}_gsx"
                ),
            ],
            axis=1,
        )
        test_rsme_gsy = pd.concat(
            [
                test_rsme_gsy,
                results.filter(
                    regex=f"rmse_test_{batch_size}_{model_name}_gsy"
                ),
            ],
            axis=1,
        )
        test_rmse_uncertainty = pd.concat(
            [
                test_rmse_uncertainty,
                results.filter(
                    regex=f"rmse_test_{batch_size}_{model_name}_uncertainty"
                ),
            ],
            axis=1,
        )
        test_rmse_idw = pd.concat(
            [
                test_rmse_idw,
                results.filter(
                    regex=f"rmse_test_{batch_size}_{model_name}_idw"
                ),
            ],
            axis=1,
        )
    
    return test_rsme_random, test_rsme_gsx, test_rsme_gsy, test_rmse_uncertainty, test_rmse_idw

def _seperate_results_val_batch(results, model = None, model_name = None, batch_sizes = None): 
    """Seperates the results dataframe into the test rmse for each selection criteria

    Args:
        results (pandas dataframe): results dataframe, create during Active Learning in batch mode
        model (model object): either model or model name needs to be provided, 
            to seperate the results accordingly, model name is preferred. 
            Defaults to None
        model_name (str, optional): model name as string matching the column 
        titles. Necessary to seperate the data Defaults to None.
        batch_sizes (list, optional): list of batch sizes to seperate the results

    Raises:
        ValueError: If neither model name nor model is provided, there is no 
        basis to identify the corresponding model string in the results
         dataframe/column names

    Returns: 
        test_rsme_random, test_rsme_gsx, test_rsme_gsy, test_rmse_uncertainty, test_rmse_idw
        Description: every object represents the test rmse for the corresponding selection criteria
    """

    val_rsme_random = pd.DataFrame()
    val_rsme_gsx = pd.DataFrame()
    val_rsme_gsy = pd.DataFrame()
    val_rmse_uncertainty = pd.DataFrame()
    val_rmse_idw = pd.DataFrame()
    if model_name == None:
        model_name = str(model).split(" ")[1]
    if model == None:
        model_name = model_name
    if (model_name is None) & (model is None):
        raise ValueError("Please provide the model or model_name")
    if batch_sizes == None:
        raise ValueError("Please provide the batch sizes")
    for batch_size in batch_sizes:
        val_rsme_random = pd.concat(
            [
                val_rsme_random,
                results.filter(regex=f"rmse_val_{batch_size}_{model_name}_random"),
            ],
            axis=1,
        )
        val_rsme_gsx = pd.concat(
            [
                val_rsme_gsx,
                results.filter(
                    regex=f"rmse_val_{batch_size}_{model_name}_gsx"
                ),
            ],
            axis=1,
        )
        val_rsme_gsy = pd.concat(
            [
                val_rsme_gsy,
                results.filter(
                    regex=f"rmse_val_{batch_size}_{model_name}_gsy"
                ),
            ],
            axis=1,
        )
        val_rmse_uncertainty = pd.concat(
            [
                val_rmse_uncertainty,
                results.filter(
                    regex=f"rmse_val_{batch_size}_{model_name}_uncertainty"
                ),
            ],
            axis=1,
        )
        val_rmse_idw = pd.concat(
            [
                val_rmse_idw,
                results.filter(
                    regex=f"rmse_val_{batch_size}_{model_name}_idw"
                ),
            ],
            axis=1,
        )
    
    return val_rsme_random, val_rsme_gsx, val_rsme_gsy, val_rmse_uncertainty, val_rmse_idw