import sys

def ActiveLearningPaths():
    """
    Set the paths for the active learning project
    :return:
    """
    # Basepath
    basepath = "../"  # Project directory
    sys.path.append(basepath)

    # Active Learning path
    AL_PATH = basepath + "04_Active_Learning/"

    # Data
    DATA_PATH = basepath + "data/"

    # Path to conda environment
    ENV_PATH = "/home/fhwn.ac.at/202375/.conda/envs/thesis/lib"

    # Resultspath
    RESULTS_PATH = AL_PATH + "results/"

    # Figure
    FIGURE_PATH = RESULTS_PATH + "figures/"

    # AL Scripts
    # AL_SCRIPTS_PATH = basepath + "al_scripts"

    # Logging
    LOG_DIR = AL_PATH + "logs/"

    # Add the paths
    sys.path.extend({DATA_PATH, FIGURE_PATH, ENV_PATH, RESULTS_PATH})
    return DATA_PATH, FIGURE_PATH, ENV_PATH, RESULTS_PATH, LOG_DIR