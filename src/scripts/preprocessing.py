### IMPORTS ###
import pandas as pd
import numpy as np
import torch

from datetime import timedelta, datetime

### CONSTANTS ###

# earliest and latest dates in our data
START_DATE = '2016-01-01 00:00:00+00:00'
END_DATE = '2023-03-03 09:00:00+00:00'

# datetime format used in our data
DATE_FORMAT = '%Y-%m-%d %H:00:00+00:00'

def fill_gaps(counter, start_date=None, end_date=None):
    """Fill in the gaps in data with -1 for roadwork (leading cause of missing counter info).

    Args:
        counter (pd.DataFrame): dataframe containing all information for specific counter (some gaps present)
        start_date (pd.datetime): day and time when we started using counters
        end_date (pd.datetime): day and time after which we have no more information

    Returns:
        fixed_counter (pd.DataFrame): copy of dataframe containing filled gaps with -1 counts.
    """

    # copy all counter information
    fixed_counter = counter.copy()

    # convert date to datetime
    fixed_counter['Date'] = pd.to_datetime(fixed_counter['Date']) 

    # if either start or end date missing then just take first/last date in counter
    if start_date is None:

        # convert to datetime format
        start_date = datetime.strptime(START_DATE, DATE_FORMAT)

    if end_date is None:
        
        # convert to datetime format
        end_date = datetime.strptime(END_DATE, DATE_FORMAT)

    # unit of measurement
    hour = timedelta(hours = 1)

    # calculate duration of time interval in units of measurement (in hours)
    daterange = int((end_date - start_date).total_seconds() / 3600)
    
    # convert date back to string (problems if its datetime)
    fixed_counter['Date'] = fixed_counter['Date'].astype(str)
    
    # set of all present timesteps (convert to string)
    datetimes = set(np.unique(counter['Date']))

    # default row (signifying missing values)
    default = pd.DataFrame({colname: [-1] for colname in counter.columns})

    # set of all timesteps
    all_datetimes = set([(start_date + i * hour).strftime(DATE_FORMAT) for i in range(daterange)])

    # set of missing timesteps
    gaps = all_datetimes - datetimes

    # create default rows (signifying missing values)
    default = pd.DataFrame()
    default['Date'] = list(gaps)

    # get all non-date columns
    counter_columns = counter.drop(['Date'], axis = 1).columns

    for colname in counter_columns:

        default[colname] = -1

    # combine our actual dataframe with values imputed for gap (values currently -1)
    fixed_counter = pd.concat([fixed_counter, default])

    # convert dates to datetime
    fixed_counter['Date'] = pd.to_datetime(fixed_counter['Date'])

    # sort data by date
    # fixed_counter = fixed_counter.sort_values(["Date"], ascending=False)

    return fixed_counter

def mark_successors_in_adj_mtx(start_node, successors_string, adj_mtx):
    successors_list = successors_string.split(',')
    if len(successors_list)==0 or successors_list[0].lower()=='nan' or successors_list[0]=='':
        return
    else:
        if start_node not in adj_mtx.index:
            print(f"WARNING: Start node {start_node} not in rows")
        
        for successor in successors_list:
            if successor not in adj_mtx.columns:
                # Successors not found among the data are some highway exits or entries 
                # or break-stops which were IGNORED on data collection!
                #print(f"Successor {successor} not in columns")
                continue
            adj_mtx.loc[start_node, successor] = 1
        return
        
def construct_edge_index(counters_aggregated : pd.DataFrame):
    all_counters = counters_aggregated['id'].unique()
    adj_mtx = pd.DataFrame(index=all_counters, columns=all_counters)
    
    counters_aggregated['successors'] = counters_aggregated['successors'].astype(str)
    all_counters = counters_aggregated['id'].unique()
    adj_mtx = pd.DataFrame(index=all_counters, columns=all_counters)

    _ = counters_aggregated[['id', 'successors']].apply(lambda x: mark_successors_in_adj_mtx(x[0], x[1], adj_mtx), axis=1)
    adj_mtx.fillna(0, inplace=True)

    n_node = len(all_counters)
    # manipulate nxn matrix into 2xnum_edges
    edge_index = torch.zeros((2, n_node**2), dtype=torch.long)
    num_edges = 0
    for i in range(n_node):
        for j in range(n_node):
            if adj_mtx.iloc[i, j] == 1:
                edge_index[0, num_edges] = i
                edge_index[1, num_edges] = j
                num_edges += 1
    # using resize_ to just keep the first num_edges entries
    edge_index = edge_index[:,:num_edges]
    return edge_index, n_node, num_edges
