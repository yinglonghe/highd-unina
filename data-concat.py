import os
import sys
#import pickle
import pandas as pd
from IPython.display import display
import multiprocessing
from tqdm import tqdm
#import datetime

from utils.read_csv import *


def data_process(i):

    idx_str = '{0:02}'.format(i)

    created_arguments = {
        'input_path': data_path + idx_str + "_tracks.csv",
        'input_static_path': data_path + idx_str + "_tracksMeta.csv",
        'input_meta_path': data_path + idx_str + "_recordingMeta.csv",
        'background_image': data_path + idx_str + "_highway.jpg",
    }

	### Read the video meta ###
    df_rec = pandas.read_csv(created_arguments["input_meta_path"])
    df_rec.to_csv(out_path + idx_str + '_meta.csv', index=False)
    df_rec.to_pickle(out_path + idx_str + '_meta.pkl')
#    meta = pd.read_pickle(out_path + idx_str + '_meta.pkl')
#    display(meta)

	### Read the static info ###
    try:
        static_info = read_static_info(created_arguments)
    except:
        print("The static info file is either missing or contains incorrect characters.")
        sys.exit(1)
    df_stc = pd.DataFrame(static_info).T
#    display(df_stc)

	### Read the track csv and convert to useful format ###
    tracks = read_track_csv(created_arguments, df_stc, df_rec)
    df_trc = pd.DataFrame(tracks)
#    display(df_trc)
	
	### Merge data ###
    df_comb = df_stc.merge(df_trc, on='id')
    df_comb.to_csv(out_path + idx_str + '_data.csv', index=False)
    df_comb.to_pickle(out_path + idx_str + '_data.pkl')
#    data = pd.read_pickle(out_path + idx_str + '_data.pkl')
#    display(data)    
    

### Global variables ###    
data_path = '../datasets/highd-dataset-v1.0/data/'
#out_path = './outputs/tracks-pkl/'
out_path = './outputs/tracks-slurm/'
os.makedirs(out_path, exist_ok=True)


if __name__ == '__main__':
    ### Step 1: Combine track and static data ###
    if False:
        cpu_cnt = multiprocessing.cpu_count()
        print(f"CPU Count: {cpu_cnt}")
        with multiprocessing.Pool(cpu_cnt) as p:
            iter = list(range(1, 61))
            r = list(tqdm(p.imap(data_process, iter), total=len(iter)))
    
    # For debug
    if True:
        for i in tqdm(range(25, 26)):
            print('Which recording is under test? '+str(i))
            data_process(i)

    ### Step 2: Combine data from different recordings ###
    if False:
        df_data = pd.DataFrame()
        df_meta = pd.DataFrame()
        for i in tqdm(range(1, 61)):
            idx_str = '{0:02}'.format(i)
            df1 = pd.read_pickle(out_path + idx_str + '_data.pkl')
            df1.insert(loc=0, column='recordingId', value=i)
            df_data = pd.concat([df_data, df1])
            
            df2 = pd.read_pickle(out_path + idx_str + '_meta.pkl')
            df_meta = pd.concat([df_meta, df2])

        df_data =  df_data.rename(columns={'id': 'vehicleId'})
        df_data.to_pickle(out_path + 'data.pkl')
        
        df_meta = df_meta.rename(columns={'id': 'recordingId'})
        df_meta.to_pickle(out_path + 'meta.pkl')
    
    ### Step 3: Load and show final data ###
    if False:
        df_meta = pd.read_pickle(out_path + 'meta.pkl')
        display(df_meta.head(5))
        
        df_data = pd.read_pickle(out_path + 'data.pkl')
        display(df_data.head(5))
