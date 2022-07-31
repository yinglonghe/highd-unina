import os
import sys
import pickle
import pandas as pd
from IPython.display import display
import multiprocessing
from tqdm import tqdm

from utils.read_csv import *


def data_process_parallel(i):
    idx_str = '{0:02}'.format(i)

    created_arguments = {
        'input_path': data_path + idx_str + "_tracks.csv",
        'input_static_path': data_path + idx_str + "_tracksMeta.csv",
        'input_meta_path': data_path + idx_str + "_recordingMeta.csv",
        'background_image': data_path + idx_str + "_highway.jpg",
    }

    
	### Read the track csv and convert to useful format ###
    tracks = read_track_csv(created_arguments)
    df1 = pd.DataFrame(tracks)
    # display(df1)

    
	### Read the static info ###
    try:
        static_info = read_static_info(created_arguments)
    except:
        print("The static info file is either missing or contains incorrect characters.")
        sys.exit(1)
    df2 = pd.DataFrame(static_info).T
    # display(df2)
    
	
	### Merge data ###
    df = df2.merge(df1, on='id')
    # df.to_csv(out_path + idx_str + '_data.csv', index=False)
    df.to_pickle(out_path + idx_str + '_data.pkl')
    data = pd.read_pickle(out_path + idx_str + '_data.pkl')
    # display(data)
    
    
	### Read the video meta ###
    try:
        meta_dictionary = read_meta_info(created_arguments)
    except:
        print("The video meta file is either missing or contains incorrect characters.")
        sys.exit(1)
        
    with open(out_path + idx_str + '_meta.pickle', 'wb') as f:
        pickle.dump(meta_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(out_path + idx_str + '_meta.pickle', 'rb') as f:
        meta = pickle.load(f)
    # display(meta)


### Global variables ###    
data_path = '../datasets/highd-dataset-v1.0/data/'
out_path = './outputs/tracks-pkl/'
os.makedirs(out_path, exist_ok=True)


if __name__ == '__main__':
    cpu_cnt = multiprocessing.cpu_count()
    print(f"CPU Count: {cpu_cnt}")
    with multiprocessing.Pool(cpu_cnt) as p:
        iter = list(range(1, 61))
        r = list(tqdm(p.imap(data_process_parallel, iter), total=len(iter)))
