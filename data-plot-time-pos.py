import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import multiprocessing
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot

### Global variables ###
indir = './outputs/tracks-slurm/'
outdir = './outputs/tracks-plt/'
os.makedirs(outdir, exist_ok=True)


if __name__ == '__main__':
    idx = 60
    idx_str = '{0:02}'.format(idx)
    
    df_meta = pd.read_pickle(indir + idx_str + '_meta.pkl')
    upperLaneMarkings = np.fromstring(df_meta['upperLaneMarkings'][0], sep=";")
    lowerLaneMarkings = np.fromstring(df_meta['lowerLaneMarkings'][0], sep=";")
    upperLaneId = list(range(2, len(upperLaneMarkings)+1))
    lowerLaneId = list(range(len(upperLaneMarkings)+2, len(upperLaneMarkings)+len(lowerLaneMarkings)+1))
    
    df = pd.read_pickle(indir + idx_str + '_data.pkl')
    
    fig1a = make_subplots(rows=len(upperLaneId), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02)
    fig1b = make_subplots(rows=1, cols=1)

    fig2a = make_subplots(rows=len(lowerLaneId), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02)
    fig2b = make_subplots(rows=1, cols=1)

    for i in tqdm(range(len(df))):
        df_traj = pd.DataFrame({
            'time': (df['frame'][i]-1)/df_meta['frameRate'][0],
            'x': df['x'][i],
            'laneId': df['laneId'][i],})
        
        if df['drivingDirection'][i] == 1:
            fig1b.add_trace(go.Scatter(x=df_traj['time'], y=df_traj['x'],
                        mode='lines',
                        name='vehId='+str(int(df['id'][i]))),
                        row=1, col=1)
            for j, laneId in enumerate(upperLaneId):
                df_lane = df_traj.loc[df_traj['laneId'] == laneId]
                if len(df_lane) != 0:
                    fig1a.add_trace(go.Scatter(x=df_lane['time'], y=df_lane['x'],
                        mode='lines',
                        name='vehId='+str(int(df['id'][i]))),
                        row=j+1, col=1)

        elif df['drivingDirection'][i] == 2:
            fig2b.add_trace(go.Scatter(x=df_traj['time'], y=df_traj['x'],
                        mode='lines',
                        name='vehId='+str(int(df['id'][i]))),
                        row=1, col=1)
            for j, laneId in enumerate(lowerLaneId):
                df_lane = df_traj.loc[df_traj['laneId'] == laneId]
                if len(df_lane) != 0:
                    fig2a.add_trace(go.Scatter(x=df_lane['time'], y=df_lane['x'],
                        mode='lines',
                        name='vehId='+str(int(df['id'][i]))),
                        row=j+1, col=1)
        
    fig1a.update_xaxes(title_text='Time [s]', row=len(upperLaneId), col=1)
    for j, laneId in enumerate(upperLaneId):
        fig1a.update_yaxes(title_text=f'Pos [m] - Lane {laneId} - Direct 1', row=j+1, col=1)
    fig1a.update_layout(showlegend=False)
    plot(fig1a, filename=outdir + idx_str + '_time_pos_direct1_lanes.html', auto_open=False)

    fig1b.update_xaxes(title_text='Time [s]', row=1, col=1)
    fig1b.update_yaxes(title_text='Pos [m] - Direct 1', row=1, col=1)
    fig1b.update_layout(showlegend=False)
    plot(fig1b, filename=outdir + idx_str + '_time_pos_direct1.html', auto_open=False)

    fig2a.update_xaxes(title_text='Time [s]', row=len(lowerLaneId), col=1)
    for j, laneId in enumerate(lowerLaneId):
        fig2a.update_yaxes(title_text=f'Pos [m] - Lane {laneId} - Direct 2', row=j+1, col=1)
    fig2a.update_layout(showlegend=False)
    plot(fig2a, filename=outdir + idx_str + '_time_pos_direct2_lanes.html', auto_open=False)
    
    fig2b.update_xaxes(title_text='Time [s]', row=1, col=1)
    fig2b.update_yaxes(title_text='Pos [m] - Direct 2', row=1, col=1)
    fig2b.update_layout(showlegend=False)
    plot(fig2b, filename=outdir + idx_str + '_time_pos_direct2.html', auto_open=False)

#    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
#    plt.savefig('test.jpg', dpi=300, bbox_inches='tight')
