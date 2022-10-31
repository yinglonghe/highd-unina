### Import packages
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import IPython
import IPython.display
import pickle


### WindowGenerator class
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


### Read and save data
def read_and_save_dataset(
    dsIdx,
    inDir, 
    caseDir, 
    classId, 
    df_des,
    col_inputs,
    input_width, 
    label_width, 
    shift,
    minFrames=200, 
    maxFrames=500,
):
    idx_str = '{0:02}'.format(dsIdx)
    windowDir = caseDir+f'ds_{input_width}_{label_width}_{shift}/'
    dsDir = windowDir+f'{idx_str}_saved_data'
    df_data = pd.read_pickle(inDir + idx_str + '_data.pkl')
    df = df_data.loc[
        (df_data['class'] == classId) & \
        (df_data['numFrames'] >= minFrames) & \
        (df_data['numFrames'] <= maxFrames)].reset_index(drop=True)
    
    df_traj = df.loc[:, col_inputs]
    w = None
    for idx, row in tqdm(df_traj.iterrows(), total=df_traj.shape[0], mininterval=5):
        df_ = pd.DataFrame()
        for col_name, col_val in row.iteritems():
            df_[col_name] = col_val
        df_ = df_[5::6]
        
        df_ = (df_ - df_des['mean']) / df_des['std']
        w_ = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, df=df_,
                             label_columns=['xAcceleration'])
        if w == None:
            w = w_
            print(w)
        else:
            w.ds = w.ds.concatenate(w_.ds)

    tf.data.experimental.save(w.ds, dsDir)
    with open(caseDir+'/column_indices.pkl', 'wb') as f:
        pickle.dump(w.column_indices, f)
    print('\n Dataset saved')
    
    return w


### Read and save data description
def read_des(
    inDir,
    caseDir,
    col_inputs,
    classId,
):
    if os.path.exists(caseDir + 'data_des.csv'):
        df_des = pd.read_csv(caseDir + 'data_des.csv', index_col=0)
        print('\n df_des loaded')
    else:
        df = pd.read_pickle(inDir + 'data.pkl')
        df = df.loc[df['class']==classId]
        flatten = {}
        for col in tqdm(col_inputs):
            arr = df[col].to_numpy()
            flatten[col] = np.concatenate(arr).ravel()
            # break
        df_flatten = pd.DataFrame.from_dict(flatten)
        df_des = df_flatten.describe().transpose()
        df_des.to_csv(caseDir + 'data_des.csv')
        print('\n df_des saved')
    return df_des
    

if __name__ == '__main__':
    inDir = './outputs/tracks-slurm/'
    outDir = './outputs/nns/'

    df_smp = pd.read_pickle(inDir + '18_data.pkl')
    df_smp = df_smp.loc[:, 'frame':'traffic_speed']
    col_full = df_smp.columns.values.tolist()
    col_drop = ['frame', 'precedingId', 'followingId', 'leftFollowingId', 'leftAlongsideId',
                'leftPrecedingId', 'rightFollowingId', 'rightAlongsideId', 'rightPrecedingId']
    col_inputs = [x for x in col_full if x not in col_drop]
    
    # 1-1-1 (single_step_window), 6-1-1 (conv_window), 24-24-1 (wide_window), 26-24-1 (wide_conv_window) 24-24-24 (multi_window)
    input_width = 8
    label_width = 8
    shift = 8

    dsIdx = int(sys.argv[1])

    classId = 0   # car-0, truck-1
    veh_class = 'car' if classId==0 else 'truck'

    caseDir = outDir+f'inputs_{len(col_inputs)}/{veh_class}/'
    os.makedirs(caseDir, exist_ok=True)
    
    df_des = read_des(
        inDir,
        caseDir,
        col_inputs,
        classId,
    )
    
    w = read_and_save_dataset(
        dsIdx,
        inDir, 
        caseDir, 
        classId, 
        df_des,
        col_inputs,
        input_width, 
        label_width, 
        shift,
    )