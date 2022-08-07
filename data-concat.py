import os
import sys
import pandas as pd
import multiprocessing
from tqdm import tqdm

from utils.read_csv import *


### Global variables ###    
data_path = '../datasets/highd-dataset-v1.0/data/'
out_path = './outputs/tracks-slurm/'
os.makedirs(out_path, exist_ok=True)


def main(i):

    idx_str = '{0:02}'.format(i)

    created_arguments = {
        'input_path': data_path + idx_str + "_tracks.csv",
        'input_static_path': data_path + idx_str + "_tracksMeta.csv",
        'input_meta_path': data_path + idx_str + "_recordingMeta.csv",
        'background_image': data_path + idx_str + "_highway.jpg",
    }

	### Read the video meta ###
    df_rec = pd.read_csv(created_arguments["input_meta_path"])
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
    tracks = read_track_csv(created_arguments, idx_str, df_stc, df_rec)
    df_trc = pd.DataFrame(tracks)
#    display(df_trc)

	### Merge data ###
    df_comb = df_stc.merge(df_trc, on='id')
    df_comb.to_csv(out_path + idx_str + '_data.csv', index=False)
    df_comb.to_pickle(out_path + idx_str + '_data.pkl')
#    data = pd.read_pickle(out_path + idx_str + '_data.pkl')
#    display(data)    


if __name__ == '__main__':
    # iter = list(range(1, 61))
    # for i in tqdm(iter):
    #     # print('Which recording is under test? '+str(i))
    #     main(i)
    TrackId = int(sys.argv[1])
    main(TrackId)