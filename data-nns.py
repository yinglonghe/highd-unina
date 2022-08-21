### Import packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import IPython
import IPython.display
import pickle
from copy import deepcopy


### Define global variables
indir = './outputs/tracks-slurm/'
outdir = './outputs/tracks-nns/'
os.makedirs(outdir, exist_ok=True)


### Package the training procedure into a function
multi_val_performance = {}
multi_performance = {}
def model_fit(model, window, max_epochs, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        verbose=2,
                        callbacks=[early_stopping])
    return history, model


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
    
    
    def get_dataset_partitions_tf(self, train_split=0.7, val_split=0.2, test_split=0.1, shuffle=True, shuffle_size=100):
        
        assert (train_split + test_split + val_split) == 1
        
        ds_size = len(list(self.ds))
        
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
    
    
if __name__ == '__main__':
    debug_break = False
    cnt_break = 300
    # Data windowing parameters
    # 1-1-1 (single_step_window), 6-1-1 (conv_window), 24-24-1 (wide_window), 26-24-1 (wide_conv_window) 24-24-24 (multi_window)
    input_width = 4
    label_width = 1
    shift = 1

    MAX_EPOCHS = 20
    
    casedir = outdir+f'{input_width}_{label_width}_{shift}'
    os.makedirs(casedir, exist_ok=True)

    dsdir = casedir+'/saved_data'
    modeldir = casedir+'/saved_model'

    print(casedir)
    
    ### Load HighD dataset
    df_meta = pd.read_pickle(indir + 'meta.pkl')
    df_data = pd.read_pickle(indir + 'data.pkl')
    
    ### Select data and pre-processing
    df_truck = df_data.loc[(df_data['class'] == 1) & (df_data['numFrames'] >= 200) & (df_data['numFrames'] <= 500)].reset_index(drop=True)
    df_car = df_data.loc[(df_data['class'] == 0) & (df_data['numFrames'] >= 200) & (df_data['numFrames'] <= 500)].reset_index(drop=True)
    df = df_car
    
    df_stat = df.loc[:, 'recordingId':'numLaneChanges']
    df_traj = df.loc[:, 'frame':'traffic_speed']
    
    ### Pre-process data
    del_cols = ['frame', 'precedingId', 'followingId', 'leftFollowingId', 'leftAlongsideId',
                'leftPrecedingId', 'rightFollowingId', 'rightAlongsideId', 'rightPrecedingId']

    if os.path.exists(casedir+'/df_flatten.pkl'):
        df_flatten = pd.read_pickle(casedir+'/df_flatten.pkl')
        print('\n df_flatten loaded!')

    else:
        df_flatten = pd.DataFrame()
        for idx, row in tqdm(df_traj.iterrows(), total=df_traj.shape[0], mininterval=5):
            df_ = pd.DataFrame()
            for col_name, col_val in row.iteritems():
                if col_name not in del_cols:
                    df_[col_name] = col_val
            df_ = df_[5::6]
            if idx == 0:
                df_flatten = df_
            else:
                df_flatten = pd.concat([df_flatten, df_])

            if debug_break and idx == cnt_break:
                break

        df_flatten.to_pickle(casedir+'/df_flatten.pkl')
        print('\n df_flatten saved!')

    df_mean = df_flatten.mean()
    df_std = df_flatten.std()
    
    ### Data windowing 
    if os.path.exists(dsdir):
        ds_load = tf.data.experimental.load(dsdir)
        with open(casedir+'/column_indices.pkl', 'rb') as f:
            cols_load = pickle.load(f)

        w = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, ds=ds_load,
                            label_columns=['xAcceleration'], column_indices=cols_load)
        print('\n Data loaded!')
    else:
        for idx, row in tqdm(df_traj.iterrows(), total=df_traj.shape[0], mininterval=10):
            df_ = pd.DataFrame()
            for col_name, col_val in row.iteritems():
                if col_name not in del_cols:
                    df_[col_name] = col_val
            df_ = df_[5::6]
            df_ = (df_ - df_mean) / df_std

            w_ = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, df=df_,
                                 label_columns=['xAcceleration'])
            if idx == 0:
                w = w_
            else:
                w.ds = w.ds.concatenate(w_.ds)

            if debug_break and idx == cnt_break:
                break

        tf.data.experimental.save(w.ds, dsdir)
        with open(casedir+'/column_indices.pkl', 'wb') as f:
            pickle.dump(w.column_indices, f)
        print('\n Data saved!')

    w.train, w.val, w.test = w.get_dataset_partitions_tf()
    
    ### Traing neural networks
    num_features = len(w.column_indices)
    OUT_STEPS = w.label_width

    if os.path.exists(modeldir+'/my_model'):
        multi_lstm_model = tf.keras.models.load_model(modeldir+'/my_model')
        print('\n Model loaded!')
    else:
        multi_lstm_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dense(OUT_STEPS*num_features,
                                  kernel_initializer=tf.initializers.zeros()),
            tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])
        multi_lstm_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
        print('\n Model created!')
        
    history, model = model_fit(multi_lstm_model, w, MAX_EPOCHS)

    # IPython.display.clear_output()

    multi_val_performance['LSTM'] = model.evaluate(w.val)
    multi_performance['LSTM'] = model.evaluate(w.test, verbose=2)

    model.summary()
    model.save(modeldir+'/my_model')
    print('\n Model saved!')
    # w.plot(model)