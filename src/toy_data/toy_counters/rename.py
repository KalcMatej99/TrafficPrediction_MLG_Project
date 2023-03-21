import glob, os
import numpy as np

def replace_name(old, new):
    with open("../toy_counters_non_temporal_aggregated_data.csv", "rt") as fin:
        with open("../toy_counters_non_temporal_aggregated_data_2.csv", "wt") as fout:
            for line in fin:
                fout.write(line.replace(old, new))
    
    os.rename("../toy_counters_non_temporal_aggregated_data_2.csv", "../toy_counters_non_temporal_aggregated_data.csv")

def get_counters():
    # Open a file: file
    file = open('../toy_counters_non_temporal_aggregated_data.csv', mode='r')
    
    # read all lines at once
    all_of_it = file.read()
    
    # close the file
    file.close()

    all_coutners = ','.join(all_of_it.replace('"', '').split('\n')).split(',')
    all_counters = list(set([counter.strip() for counter in all_coutners]))
    all_counters = sorted(all_counters)[1:-2]

    return all_counters


linked_names = get_counters()

file_list = [old for old in glob.glob('*.csv')]

print(len(file_list))

counter_list = list(set([old.split('.')[0] for old in glob.glob('*.csv')]).union(set(linked_names)))

names = np.arange(len(counter_list))

np.random.shuffle(names)

counter_map = {counter: id for counter, id in zip(counter_list, names)}

for old, id in zip(file_list, names):
    new = f'{id}.csv'

    os.rename(old, new)

for name in counter_map.keys():
    replace_name(name, str(counter_map[name]))