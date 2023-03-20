#################################
############ IMPORTS ############
#################################

# for scoring function calculation
import torch
import numpy as np
# reading tensorboard logs
from tensorboard.backend.event_processing import event_accumulator

# save log info
import pandas as pd

# plot logs
import matplotlib.pyplot as plt

# save all prediction as list of 3D matrices (nodes x car types x timesteps)
import pickle as pkl
from os.path import exists
from datetime import datetime

# save all prediction as list of 3D matrices (nodes x car types x timesteps)
import pickle as pkl
from os.path import exists
from datetime import datetime

#################################
####### SCORING FUNCTIONS #######
#################################


def z_score(x, mean, std):
    return (x - mean) / std


def un_z_score(x_normed, mean, std):
    return x_normed * std + mean


def MAPE(v, v_):
    return torch.mean(torch.abs((v_ - v)) / (v + 1e-15) * 100)


def RMSE(v, v_):
    missing_data_indices = (v == -1)
    v_[missing_data_indices] = 0
    v[missing_data_indices] = 0
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))


#################################
####### LOGGING FUNCTIONS #######
#################################

def read_logs(path, scalars, save_dir, rounding=0, aggregate=True):
    """Read tensorboard log file and plot graphs for specified 
    variables (scalars) then save the graphs to save dir. 

    Args:
        path (str): path to log files from model
        scalars (list[str]): list of variables of interest
        save_dir (str): directory where to save graphs (.png format)
        rounding (int, optional): how much difference actually matters. Defaults to 0.
        aggregate (bool, optional): calc the means for each step (logging seems to be on a batch basis). Defaults to True.
    Returns:
        all_data(pd.DataFrame): logs in df format
    """

    # create save paths
    df_save_path = f'{save_dir}/logs.csv'

    # instantiate reader object
    ea = ea = event_accumulator.EventAccumulator(path)

    # read tensorboard logs
    ea.Reload()

    # list of (saved) directory keys
    tags = ea.Tags()

    print(tags)

    # get available variables
    available_vars = tags['scalars']

    # initialize dataframe
    all_data = pd.DataFrame()

    # read info for each measured variable
    for scalar in scalars:

        if scalar in available_vars:

            info = ea.Scalars(scalar)

            # convert to dataframe
            data = pd.DataFrame(info)

            # set name to log name
            data = data.rename(columns={'value': scalar})

            # join all data into same dataframe
            if all_data.empty:

                all_data = data

            else:

                # keep only the logs column
                data.drop(['wall_time'], axis=1, inplace=True)

                # add only one column of interest (use merge in case logging happens at different timesteps)
                all_data = all_data.merge(data, on='step', how='outer')

                # round results
                all_data = all_data.round(rounding)

            # drop duplicates (seem to be generated somehow - double logging not really that big a deal)
            all_data.drop_duplicates(inplace=True)

            if aggregate:
                all_data = all_data.groupby(['step'], as_index=False).mean()

            # save dataframe
            all_data.to_csv(df_save_path, index=False)

        else:

            print(f'{scalar} variable was not logged.')

    return all_data


def plot_logs(data, graph_list, save_paths, aggregate=True):
    """Plot and save relevant graphs from data (logs).

    Args:
        data (pd.DataFrame): dataframe of logs
        graph_list (list[list[str]]): list of graph (variables) to plot/save
        save_paths (str): paths where to save graphs (.png format)
        aggregate (bool, optional): calc the means for each step (logging seems to be on a batch basis). Defaults to True.
    """

    for scalars, save_path in zip(graph_list, save_paths):

        # select displayed variables
        selection = ['step'] + scalars
        graph_data = data[selection]

        # don't display missing rows
        graph_data = graph_data.dropna()

        if aggregate:
            graph_data = graph_data.groupby(['step'], as_index=False).mean()

        # create plot
        ax = plt.gca()

        for scalar in scalars:

            # lineplot for each variable
            graph_data.plot(kind='line', x='step', y=scalar, ax=ax)
        
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.savefig(save_path, dpi=100)
        

        # destroy plot
        plt.cla()

####################################################
####### VISUALIZE EACH INDIVIDUAL PREDICTION #######
####################################################


def generate_savefile(name):
    """Generate names for save files (TODO)

    Args:
        name (str): some distinguishing savefile info

    Returns:
        unique_name (str): added time date
    """

    current_datetime = datetime.today().strftime('%Y-%m-%dT%H-%M-%S')

    unique_name = f'{name}-{current_datetime}'

    return unique_name


def save_all_predictions(y_pred, y_true, dim_vals, save_directory):
    """Save all prediction as list tuples (y_pred, y_true) where both are 3D matrices (n_graphs x nodes x timesteps).

    Args:
        y_pred (torch.Tensor): model predictions
        y_true (torch.Tensor): actual values
        dim_vals (list[list[str]]): 2x List of counters and timesteps
        save_directory (str): location where to save at or if exists list of results
    """
    save_file = save_directory + '/ygt_ypred.pkl'
    # create new list
    res = []

    # not the first batch
    if exists(save_file):

        # load the previously saved list (simply append current predictions)
        with open(save_file, 'rb') as handle:
            err = pkl.load(handle)

    # append current predictions
    res.append((y_pred, y_true, dim_vals))

    # store results
    with open(save_file, 'wb') as handle:
        pkl.dump(res, handle, protocol=pkl.HIGHEST_PROTOCOL)


def plot_predictions_vs_gt_worst(y_pred=None, y_true=None, length=168, maximum_search_space = 1000000, pickle_path=None):
    """Plot worst prediction.

    Args:
        y_pred (torch.Tensor): model predictions
        y_true (torch.Tensor): actual values
        length (integer): Size of prediction
        maximum_search_space (integer): Size of search space
        pickle_path (str): location where to find predictions and gt
    """
    if pickle_path is not None:
        with open(pickle_path, 'rb') as f:
            data = pkl.load(f)
            y_pred = data[0][0].reshape(-1)
            y_true = data[0][1].reshape(-1)
    else:
        y_pred = y_pred[0][0].reshape(-1)
        y_true = y_true[0][1].reshape(-1)
    ####################################################
####### VISUALIZE EACH INDIVIDUAL PREDICTION #######
####################################################

def generate_savefile(name):
    """Generate names for save files (TODO)

    Args:
        name (str): some distinguishing savefile info

    Returns:
        unique_name (str): added time date
    """

    current_datetime = datetime.today().strftime('%Y-%m-%dT%H-%M-%S')

    unique_name = f'{name}-{current_datetime}'

    return unique_name


def save_all_predictions(y_pred, y_true, dim_vals, save_directory):
    """Save all prediction as list tuples (y_pred, y_true) where both are 3D matrices (n_graphs x nodes x timesteps).

    Args:
        y_pred (torch.Tensor): model predictions
        y_true (torch.Tensor): actual values
        dim_vals (list[list[str]]): 2x List of counters and timesteps
        save_directory (str): location where to save at or if exists list of results
    """
    save_file = save_directory + '/ygt_ypred.pkl' 
    # create new list
    res = []

    # not the first batch
    if exists(save_file):

        # load the previously saved list (simply append current predictions)
        with open(save_file, 'rb') as handle:
            err = pkl.load(handle)

    # append current predictions
    res.append((y_pred, y_true, dim_vals))

    # store results
    with open(save_file, 'wb') as handle:
        pkl.dump(res, handle, protocol=pkl.HIGHEST_PROTOCOL)

def plot_predictions_vs_gt(y_pred=None, y_true=None, start=0, stop=1000, pickle_path=None):
  if pickle_path is not None:
      with open(pickle_path, 'rb') as f:
          data = pkl.load(f)
          y_pred = data[0][0].reshape(-1)[start:stop]
          y_true = data[0][1].reshape(-1)[start:stop]
  else:
      y_pred = y_pred[0][0].reshape(-1)[start:stop]
      y_true = y_true[0][1].reshape(-1)[start:stop]

  rng = stop-start
  plt.style.use('seaborn-darkgrid')
  plt.plot(range(rng), y_pred, label="pred")
  plt.plot(range(rng), y_true, label="gt")
  plt.vlines([i*168 for i in range(rng//168+1)], ymin=0, ymax=3000, linestyles='dashed', colors="black")
  plt.legend()

    worst_mse = 0
    best_index = 0
    for index in range(maximum_search_space):
        error_mse = np.average((y_pred[index: index + length] - y_true[index: index + length]) ** 2)
        if error_mse > worst_mse:
            worst_mse = error_mse
            best_index = index
    y_pred = y_pred[best_index: best_index + length]
    y_true = y_true[best_index: best_index + length]
    print(best_index)
    rng = length
    plt.style.use('seaborn-darkgrid')
    plt.plot(range(rng), y_true, label="Ground Truth")
    plt.plot(range(rng), y_pred, label="Prediction")
    plt.plot(range(rng), np.abs(y_true - y_pred), label="Absolute Error")
    plt.xlabel("Time")
    plt.ylabel("Number of cars")
    plt.title("Comparison between unsuccessful prediction and ground truth")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.legend()

def plot_predictions_vs_gt_best(y_pred=None, y_true=None, length=168, maximum_search_space = 1000000, pickle_path=None):
    """Plot worst prediction.

    Args:
        y_pred (torch.Tensor): model predictions
        y_true (torch.Tensor): actual values
        length (integer): Size of prediction
        maximum_search_space (integer): Size of search space
        pickle_path (str): location where to find predictions and gt
    """
    if pickle_path is not None:
        with open(pickle_path, 'rb') as f:
            data = pkl.load(f)
            y_pred = data[0][0].reshape(-1)
            y_true = data[0][1].reshape(-1)
    else:
        y_pred = y_pred[0][0].reshape(-1)
        y_true = y_true[0][1].reshape(-1)
    
    best_mse = 999999999999999
    best_index = 0
    for index in range(maximum_search_space):
        error_mse = np.average((y_pred[index: index + length] - y_true[index: index + length]) ** 2)
        if error_mse < best_mse:
            best_mse = error_mse
            best_index = index
    y_pred = y_pred[best_index: best_index + length]
    y_true = y_true[best_index: best_index + length]
    print(best_index)
    rng = length
    plt.style.use('seaborn-darkgrid')
    plt.plot(range(rng), y_true, label="Ground Truth")
    plt.plot(range(rng), y_pred, label="Prediction")
    plt.plot(range(rng), np.abs(y_true - y_pred), label="Absolute Error")
    plt.xlabel("Time")
    plt.ylabel("Number of cars")
    plt.title("Comparison between prediction and ground truth")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.legend()


def plot_predictions_vs_gt(y_pred=None, y_true=None, start=0, stop=1000, pickle_path=None):
    """Plot comparison between predictions and gt.

    Args:
        y_pred (torch.Tensor): model predictions
        y_true (torch.Tensor): actual values
        start (integer): Start index
        stop (integer): Stop index
        pickle_path (str): location where to find predictions and gt
    """
    if pickle_path is not None:
        with open(pickle_path, 'rb') as f:
            data = pkl.load(f)
            y_pred = data[0][0].reshape(-1)[start:stop]
            y_true = data[0][1].reshape(-1)[start:stop]
    else:
        y_pred = y_pred[0][0].reshape(-1)[start:stop]
        y_true = y_true[0][1].reshape(-1)[start:stop]

    rng = stop-start
    plt.style.use('seaborn-darkgrid')
    plt.plot(range(rng), y_true, label="Ground Truth")
    plt.plot(range(rng), y_pred, label="Prediction")
    plt.plot(range(rng), np.abs(y_true - y_pred), label="Absolute Error")
    plt.xlabel("Time")
    plt.ylabel("Number of cars")
    plt.title("Comparison between prediction and ground truth for Counter 0011")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.vlines([i*168 for i in range(rng//168+1)], ymin=0, ymax=3000, linestyles='dashed', colors="black")
    plt.legend()


if __name__ == '__main__':

    # which logs you want to read
    tb_path = '/home/bro/Documents/FRI/MLG/project/TrafficPrediction_MLG_Project/src/runs/Mar06_21-28-48_bro-MS-7C91/events.out.tfevents.1678134528.bro-MS-7C91.24502.5'
    
    # where to save df
    save_path = '/home/bro/Documents/FRI/MLG/project/TrafficPrediction_MLG_Project/src/runs/Mar06_21-28-48_bro-MS-7C91'

    # what you want to read from logs
    logs = read_logs(tb_path, ['Loss/train', 'MAE/train', 'RMSE/train',
                               'MAE/val', 'RMSE/val'], save_path)

    # put where you want to save graphs
    graph_paths = [
        '/home/bro/Documents/FRI/MLG/project/TrafficPrediction_MLG_Project/src/runs/Mar06_21-28-48_bro-MS-7C91/loss.png',
        '/home/bro/Documents/FRI/MLG/project/TrafficPrediction_MLG_Project/src/runs/Mar06_21-28-48_bro-MS-7C91/rmse.png',
        '/home/bro/Documents/FRI/MLG/project/TrafficPrediction_MLG_Project/src/runs/Mar06_21-28-48_bro-MS-7C91/mape.png',
        '/home/bro/Documents/FRI/MLG/project/TrafficPrediction_MLG_Project/src/runs/Mar06_21-28-48_bro-MS-7C91/mae.png'
    ]

    # which graphs to plot
    plot_logs(logs, [['Loss/train'],
                     ['RMSE/train', 'RMSE/val'],
                     ['MAE/val', 'MAE/train']
                     ], graph_paths)
