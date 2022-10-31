import os, sys
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import pickle

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


indir = './outputs/tracks-slurm/'
outdir = './outputs/unina-cf-nns-norm_min_max/'

df_meta = pd.read_pickle(indir + 'meta.pkl')
recIdSet = list(df_meta.loc[df_meta['locationId']==2, 'recordingId'].unique())


df = pd.DataFrame()
classId = 0 # car-0, truck-1
veh_class = 'car' if classId==0 else 'truck'

for i in recIdSet:
    idx_str = '{0:02}'.format(i)
    df_ = pd.read_pickle(indir + idx_str + '_data.pkl')
    df_.insert(loc=0, column='recordingId', value=i)
    df = pd.concat([df, df_])
df =  df.rename(columns={'id': 'vehicleId'})
df = df.loc[
    (df['class'] == classId) &   # 0-car, 1-truck
    (df['numFrames'] >= 200) & 
    (df['numFrames'] <= 500)].reset_index(drop=True)
# df.to_pickle(outdir + 'data.pkl')


df_traj = df.loc[:, 'frame':'traffic_speed']
col_full = df_traj.columns.values.tolist()

col_drop = ['frame', 'precedingId', 'followingId', 'leftFollowingId', 'leftAlongsideId',
            'leftPrecedingId', 'rightFollowingId', 'rightAlongsideId', 'rightPrecedingId']
# Drop lateral information
col_drop += ['y', 'yVelocity', 'yAcceleration', 'laneId', 
             'Left_Pre_X', 'Left_Pre_Speed', 'Left_Al_X', 'Left_Al_Speed', 'Left_Fol_X', 'Left_Fol_Speed',
             'Right_Pre_X', 'Right_Pre_Speed', 'Right_Al_X', 'Right_Al_Speed', 'Right_Fol_X', 'Right_Fol_Speed']
# Drop traffic information
col_drop += ['traffic_density', 'traffic_speed']

col_inputs = [x for x in col_full if x not in col_drop]


flatten = {}
for col in col_inputs:
    arr = df_traj[col].to_numpy()
    flatten[col] = np.concatenate(arr).ravel()
df_flatten = pd.DataFrame.from_dict(flatten)
df_des = df_flatten.describe().transpose()
scndir = outdir + f'{len(col_inputs)}-{veh_class}/'
os.makedirs(scndir, exist_ok=True)
df_des.to_csv(scndir + 'data_des.csv')


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, batch_size=32,
                 df=None, ds=None, label_columns=None, column_indices=None):
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        # Store the raw data.
        if df is None:
            assert ds is not None and column_indices is not None
            self.ds = ds
            self.column_indices = column_indices
        else:
            self.df = df
            self.column_indices = {name: i for i, name in
                       enumerate(self.df.columns)}
            self.ds = self.make_dataset(self.df)
    
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds
    
    
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    
    def plot(self, model=None, plot_col='xAcceleration', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Step')
    
    
    def get_dataset_partitions_tf(self, ds_size, train_split=0.7, val_split=0.2, test_split=0.1, shuffle=True, shuffle_size=1000):
        
        assert (train_split + test_split + val_split) == 1
        
        # ds_size = len(list(self.ds))
        
        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = self.ds.shuffle(shuffle_size, seed=12)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train = ds.take(train_size)
        val = ds.skip(train_size).take(val_size)
        test = ds.skip(train_size).skip(val_size)

        return train, val, test
    
    
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.ds))
            # And cache it for next time
            self._example = result
        return result
    

# 10_10_10, 10_5_5, 5_1_1, 1_1_1
input_width = 10
label_width = 10
shift = 10

casedir = outdir+f'{len(col_inputs)}-{veh_class}/{input_width}_{label_width}_{shift}/'
dsdir = casedir+'saved_data'
print(casedir)

df_traj = df.loc[:, col_inputs]
w = None
ds_size = 0
for idx, row in tqdm(df_traj.iterrows(), total=df_traj.shape[0], mininterval=5):
    df_ = pd.DataFrame()
    for col_name, col_val in row.iteritems():
        df_[col_name] = col_val
    df_ = df_[4::5]

    df_ = (df_ - df_des['mean']) / df_des['std']
    # df_ = (df_ - df_des['min']) / (df_des['max'] - df_des['min'])
    w_ = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, df=df_,
                         label_columns=['xAcceleration'])
    ds_size += len(list(w_.ds))
    if w == None:
        w = w_
    else:
        w.ds = w.ds.concatenate(w_.ds)
    
    # if idx==100:
    #     break

print(w)
tf.data.experimental.save(w.ds, dsdir)
with open(casedir+'/ds_size.txt', 'w') as f:
    f.write('%d' % ds_size)
with open(casedir+'/column_indices.pkl', 'wb') as f:
    pickle.dump(w.column_indices, f)
print('\n Data saved!')