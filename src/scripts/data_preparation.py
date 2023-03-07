### IMPORTS ###
import pandas as pd
import numpy as np
import torch
import glob
from torch_geometric.data import Data
import logging
import datetime
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


def add_hours_to_holidays(holidays):
    """Original holidays specified for days - but our task requires hourly predictions.
    Create dataframe, in which, each hour is specified as a holiday.

    Args:
        holidays (pd.DataFrame): holidays for several countries

    Returns:
        holiday_markers (pd.DataFrame): same as holidays but each hour is annotated
    """
    
    # make copy of holiday dataframe (prevent changes)
    holidays = holidays.copy()
    
    # marking for each holiday (last from 0:0 - 24:00)
    holiday_markers = pd.DataFrame()
    
    holidays['Date'] = pd.to_datetime(holidays['Date'])
    
    # unit of measurement
    hour = timedelta(hours = 1)
    
    for i in range(23):
        
        # holidays at a specific hour of the day
        holidays = holidays.copy()
        
        # shift holiday from midnight to specific hour
        holidays['Date'] += hour
        
        # add to all holiday markers
        holiday_markers = pd.concat([holiday_markers, holidays])
        
    # specify one datetime format (for combining with other datasets)
    holiday_markers['Date'] = holiday_markers['Date'].dt.strftime(DATE_FORMAT)
        
    return holiday_markers


def mark_holidays(counters, holiday_markers, shift = 7, counties_of_interest = ['Slovenia'], holiday_types = [False, True]):
    """Prepend each counter with info whether there'll be a holiday in the next x days (shift).
    This way when we join we get a feature of 7x1 specifying holidays in next week.

    Args:
        counters (pd.DataFrame): without added holiday feature
        holiday_markers (pd.DataFrame): marked holidays by hours
        shift (int, optional): how many day into the future we want. Defaults to 7.
        counties_of_interest (list, optional): for which countries. Defaults to ['Slovenia'].
        holiday_types (list, optional): unpaid and/or paid. Defaults to [False, True].

    Returns:
        annot_counters(pd.DataFrame): with added holiday feature
    """
    
    # dataframe counter (with marked holidays)
    annot_counters = counters.copy().drop_duplicates()
    
    # unit of shift
    day = timedelta(days = 1)
    
    # shift holiday markers into the past (so they reflect future holidays)
    # print(list(holiday_markers['Date'].unique()))
    holiday_markers['Date'] = pd.to_datetime(holiday_markers['Date'])
    holiday_markers['Date'] -= shift * day
    holiday_markers['Date'] = holiday_markers['Date'].dt.strftime(DATE_FORMAT)

    # change counter dates to strings
    annot_counters['Date'] = annot_counters['Date'].dt.strftime(DATE_FORMAT)

    for country in counties_of_interest:
        
        for is_paid in holiday_types:
        
            # get holidays specific to country (ignore name and type - prob. to noisy info)
            country_holidays = holiday_markers.loc[(holiday_markers['country'] == country) & (holiday_markers['isPaid'] == is_paid)][['Date', 'isPaid']]
            
            # drop any possibly duplicated columns
            country_holidays = country_holidays.drop_duplicates()

            # feature name 
            feat_name = f'{"Paid" if is_paid else "Unpaid"} Holiday-{country}'
            
            # rename is holiday column to distinguish between different countries
            country_holidays.rename(columns = {'isPaid': feat_name}, inplace = True)

            # set all values to true this will be the feature vector
            country_holidays[feat_name] = True
            
            # append holiday info
            annot_counters = annot_counters.merge(country_holidays, on = 'Date', how = 'left')
    
    # all rows that are not holidays should be marked as false
    annot_counters = annot_counters.fillna(False)
    
    return annot_counters


def mark_successors_in_adj_mtx(start_node, successors_string, adj_mtx):
    """In the adj_mtx DataFrame, mark all the successor nodes
    of start_node which are stored in successors_string 

    Args:
        start_node (str): the counter for which we want to mark his successors (ex: 0111-1)
        successors_string (str): the successor nodes for start_node (ex: '0112-2,0231-1')
        adj_mtx (pd.datetime): adjacency matrix of size [n,n]

    Returns:
        void : nothing is returned, only adj_mtx is changed
    """
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
    """Given non-temporal data about all counters, construct the edge_index 
    array needed for PyG training

    Args:
        counters_aggregated (pd.DataFrame): data-frame containing counters data in each row

    Returns:
        edge_index (np.array): nothing is returned, only adj_mtx is changed
        n_node (int): number of nodes
        num_edges (int): number of edges
    """
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

def number_of_countries_in_holiday_dataset(config):
    holidays = pd.read_csv(config["holidays_path"])
    return len(set(holidays["country"]))

def prepare_holidays_dataset(config):
    """TODO

    Args:
        config (json): parameter dictionary

    Returns:
        holiday_markers (pd.DataFrame): clean dataframe with holidays
    """
    # holidays for each region
    holidays = pd.read_csv(config["holidays_path"])

    # drop holiday region (all values look like NaN - all we have is Austrian regions or very german sounding Slovenian regions)
    holidays = holidays.drop(['region'], axis = 1)

    # remove NaNs
    holidays = holidays.drop_duplicates()

    # name holidays properly
    holidays.rename(columns = {'date': 'Date'}, inplace = True)

    holiday_markers = add_hours_to_holidays(holidays)
    holiday_markers["Date"] = pd.to_datetime(holiday_markers['Date'])
    return holiday_markers


def prepare_historical_dataset(config):
    """TODO

    Args:
        config (json): parameter dictionary

    Returns:
        slovenian_holiday_markers (pd.DataFrame): clean dataframe with holidays
    """
    counters_df = pd.DataFrame()
    for fname in glob.glob(config["counter_files_path"] + "*.csv"):
        counter_data = pd.read_csv(fname)
        counter_data = fill_gaps(counter_data)
        #counter_data = mark_holidays(counter_data, holiday_markers)
        counter_data['Date'] = pd.to_datetime(counter_data['Date']) 
        counter_data.index = counter_data['Date']
        counter_data = counter_data.sort_index(ascending=False)
        # We don't need to work with all past data.
        # Select enough data points to extract N_GRAPHS with F_IN and F_OUT timepoints
        
        counter_data = counter_data.iloc[0:(config["F_IN"]+config["F_OUT"]+config["N_GRAPHS"]), :]
        counter_id = fname.split('/')[-1].split('.csv')[0]

        if counters_df.empty:
            counters_df = pd.DataFrame(counter_data[config['target_col']])
            counters_df.columns = [counter_id]
        else:
            columns = list(counters_df.columns) + [counter_id]
            counters_df = pd.concat([counters_df, counter_data[config['target_col']]], axis=1)
            counters_df.columns = columns 

    return counters_df

def prepare_pyg_dataset(config):
    """Construct a list which contains a Data instance in every element

    Args:
        config (json): parameter dictionary

    Returns:
        dataset (list): dataset containing Data instances
    """
    logging.info("Preparing data...")

    # Prepare dataset with holiday data
    if config["USE_HOLIDAY_FEATURES"]:
        holiday_markers = prepare_holidays_dataset(config)
        logging.info("Holiday features successfully prepared")

    # Prepare dataset with historical counter data
    counters_df = prepare_historical_dataset(config)
    logging.info("Historical counter data successfully read")

    #Prepare edge_index matrix
    counters_aggregated = pd.read_csv(config['counters_nontemporal_aggregated'])
    edge_index, n_node, _ = construct_edge_index(counters_aggregated)
    logging.info("Edge index constructed")

    #Prepare matrices X shaped:[N_GRAPHS, N_NODES, F_IN] and Y shaped:[N_GRAPHS, N_NODES, F_OUT] 
    final_dataset = []
    for i in range(1, config["N_GRAPHS"]+1):
        g = Data()
        g.__num_nodes__ = n_node
        g.edge_index = edge_index
        train_test_chunk = counters_df.iloc[(-i-(config['F_IN']+config['F_OUT'])):(-i),:]
        
        X = train_test_chunk.iloc[:config['F_IN'],:].to_numpy().T
        Y = train_test_chunk.iloc[config['F_IN']:,:].to_numpy().T

        if config["USE_HOLIDAY_FEATURES"]:
            current_date = np.max(train_test_chunk.iloc[:config['F_IN'],:].index)
            for country in list(set(holiday_markers["country"])):
                selected_country_holiday_markers = holiday_markers[holiday_markers["country"] == country]
                for day_index in range(1, 8):
                    if len(selected_country_holiday_markers[selected_country_holiday_markers["Date"] == current_date + timedelta(days=day_index)].index) > 0:
                        X = np.hstack((X, np.ones((len(X), 1))))
                    else:
                        X = np.hstack((X, np.zeros((len(X), 1))))

        g.x = torch.FloatTensor(X)
        g.y = torch.FloatTensor(Y)
        final_dataset += [g]

    logging.info("Final dataset constructed")
    return final_dataset    

    

def split_dataset(dataset, config):
    """Split the given dataset into train, validation and test

    Args:
        dataset (list): the dataset we wish to split

    Returns:
        train_g (list): train dataset,
        val_g (list): validation dataset,
        test_g (list): test dataset,
    """
    split_train, split_val, _ = config['TRAIN_TEST_PROPORTION' ]
    index_train = int(np.floor(config["N_GRAPHS"]*split_train))
    index_val = int(index_train + np.floor(config["N_GRAPHS"]*split_val))
    train_g = dataset[:index_train]
    val_g = dataset[index_train:index_val]
    test_g = dataset[index_val:]

    print("Size of train data:", len(train_g))
    print("Size of validation data:", len(val_g))
    print("Size of test data:", len(test_g))

    logging.info("Dataset splitted to train,val,test")
    return train_g, val_g, test_g

