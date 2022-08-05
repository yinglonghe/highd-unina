import os
import pandas
import numpy as np
from IPython.display import display
import multiprocessing
from itertools import product
from functools import partial
from tqdm import tqdm


# TRACK FILE
BBOX = "bbox"
FRAMES = "frames"
FRAME = "frame"
TRACK_ID = "id"
X = "x"
Y = "y"
WIDTH = "width"
HEIGHT = "height"
X_VELOCITY = "xVelocity"
Y_VELOCITY = "yVelocity"
X_ACCELERATION = "xAcceleration"
Y_ACCELERATION = "yAcceleration"
FRONT_SIGHT_DISTANCE = "frontSightDistance"
BACK_SIGHT_DISTANCE = "backSightDistance"
DHW = "dhw"
THW = "thw"
TTC = "ttc"
PRECEDING_X_VELOCITY = "precedingXVelocity"
PRECEDING_ID = "precedingId"
FOLLOWING_ID = "followingId"
LEFT_PRECEDING_ID = "leftPrecedingId"
LEFT_ALONGSIDE_ID = "leftAlongsideId"
LEFT_FOLLOWING_ID = "leftFollowingId"
RIGHT_PRECEDING_ID = "rightPrecedingId"
RIGHT_ALONGSIDE_ID = "rightAlongsideId"
RIGHT_FOLLOWING_ID = "rightFollowingId"
LANE_ID = "laneId"

# STATIC FILE
INITIAL_FRAME = "initialFrame"
FINAL_FRAME = "finalFrame"
NUM_FRAMES = "numFrames"
CLASS = "class"
DRIVING_DIRECTION = "drivingDirection"
TRAVELED_DISTANCE = "traveledDistance"
MIN_X_VELOCITY = "minXVelocity"
MAX_X_VELOCITY = "maxXVelocity"
MEAN_X_VELOCITY = "meanXVelocity"
MIN_DHW = "minDHW"
MIN_THW = "minTHW"
MIN_TTC = "minTTC"
NUMBER_LANE_CHANGES = "numLaneChanges"

# VIDEO META
ID = "id"
FRAME_RATE = "frameRate"
LOCATION_ID = "locationId"
SPEED_LIMIT = "speedLimit"
MONTH = "month"
WEEKDAY = "weekDay"
START_TIME = "startTime"
DURATION = "duration"
TOTAL_DRIVEN_DISTANCE = "totalDrivenDistance"
TOTAL_DRIVEN_TIME = "totalDrivenTime"
N_VEHICLES = "numVehicles"
N_CARS = "numCars"
N_TRUCKS = "numTrucks"
UPPER_LANE_MARKINGS = "upperLaneMarkings"
LOWER_LANE_MARKINGS = "lowerLaneMarkings"

os.makedirs('test', exist_ok=True)
def process_track_data(all_track_df, df_stc, df_rec):

    L = 0.424  # length of the location under study (in km)
    upperLaneMarkings = np.fromstring(df_rec['upperLaneMarkings'][0], sep=";")
    lowerLaneMarkings = np.fromstring(df_rec['lowerLaneMarkings'][0], sep=";")
    upperLaneId = list(range(2, len(upperLaneMarkings)+1))
    lowerLaneId = list(range(len(upperLaneMarkings)+2, len(upperLaneMarkings)+len(lowerLaneMarkings)+1))
    nearby_index = [2,6]

    #loop through the tracksMeta line-by-line, each line is a vehicle
    df_stc.reset_index(drop=True, inplace=True)

    df = pandas.DataFrame()
    for l in tqdm(range(0,len(df_stc.index)), mininterval=120):
#    for l in range(0,len(df_stc.index)):
        arr_new = []
        trackID = df_stc["id"][l]
        drivingDirection = df_stc["drivingDirection"][l]  #1 for upper lanes (drive to the left), and 2 for lower lanes (drive to the right)

        track_df = all_track_df[all_track_df["id"]==trackID].reset_index(drop=True)
        if drivingDirection==1:
            track_df["xVelocity"]=-track_df["xVelocity"]
            track_df["xAcceleration"]=-track_df["xAcceleration"]
            track_df["precedingXVelocity"]=-track_df["precedingXVelocity"]

        # loop through each line in the track data
        for t in range(0,len(track_df.index)):
            frameID = track_df['frame'][t]

            # Find traffic-related variables: Density and traffic mean speed
            if drivingDirection==1:
                traffic_density = len(all_track_df[(all_track_df["frame"]==frameID) & (all_track_df["laneId"].isin(upperLaneId))]) / (L*len(upperLaneId))
            else: traffic_density = len(all_track_df[(all_track_df["frame"]==frameID) & (all_track_df["laneId"].isin(lowerLaneId))]) / (L*len(lowerLaneId))
            
            if drivingDirection==1:
                traffic_speed = -np.mean(all_track_df.loc[(all_track_df["frame"]==frameID) & (all_track_df["laneId"].isin(upperLaneId)),"xVelocity"])
            else: traffic_speed = np.mean(all_track_df.loc[(all_track_df["frame"]==frameID) & (all_track_df["laneId"].isin(lowerLaneId)),"xVelocity"])
            
            # Now look at the all_track_df data to find the location and speed of surrounding vehicles for each vehicle we keep [x_location,speed]
            if track_df["leftPrecedingId"][t]!=0:
                leftPreceding_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["leftPrecedingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                leftPreceding_df[0] = np.abs(leftPreceding_df[0]-track_df["x"][t])
                leftPreceding_df[1]= np.abs(leftPreceding_df[1])
            else: leftPreceding_df = np.array([0,0])

            if track_df["leftFollowingId"][t]!=0:
                leftFollowing_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["leftFollowingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                leftFollowing_df[0] = np.abs(leftFollowing_df[0]-track_df["x"][t])
                leftFollowing_df[1] = np.abs(leftFollowing_df[1])
            else: leftFollowing_df = np.array([0,0])

            if track_df["leftAlongsideId"][t]!=0:
                leftAlongside_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["leftAlongsideId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                leftAlongside_df[0] = np.abs(leftAlongside_df[0]-track_df["x"][t])
                leftAlongside_df[1] = np.abs(leftAlongside_df[1])
            else: leftAlongside_df = np.array([0,0])
            

            if track_df["rightPrecedingId"][t]!=0:
                rightPreceding_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["rightPrecedingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                rightPreceding_df[0] = np.abs(rightPreceding_df[0]-track_df["x"][t])
                rightPreceding_df[1] = np.abs(rightPreceding_df[1])
            else: rightPreceding_df = np.array([0,0])

            if track_df["rightAlongsideId"][t]!=0:
                rightAlongside_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["rightAlongsideId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                rightAlongside_df[0] = np.abs(rightAlongside_df[0]-track_df["x"][t]) 
                rightAlongside_df[1] = np.abs(rightAlongside_df[1])
            else: rightAlongside_df = np.array([0,0])

            if track_df["rightFollowingId"][t]!=0:
                rightFollowing_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["rightFollowingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                rightFollowing_df[0] = np.abs(rightFollowing_df[0]-track_df["x"][t])
                rightFollowing_df[1] = np.abs(rightFollowing_df[1])
            else: rightFollowing_df = np.array([0,0])
            
            #The output of the car-following model is the acceleration
#            Acceleration = np.array(track_df.loc[t+1,"xAcceleration"])
            # Combine the whole line of data
            line_new = np.hstack([frameID, leftPreceding_df, leftAlongside_df, leftFollowing_df, rightPreceding_df, rightAlongside_df, rightFollowing_df, traffic_density, traffic_speed])
            
            arr_new.append(line_new)

        df_new = pandas.DataFrame(arr_new, 
            columns =  ['frame', 'Left_Pre_X', 'Left_Pre_Speed', 'Left_Al_X', 'Left_Al_Speed', 'Left_Fol_X', 'Left_Fol_Speed',
                'Right_Pre_X', 'Right_Pre_Speed', 'Right_Al_X', 'Right_Al_Speed', 'Right_Fol_X', 'Right_Fol_Speed',
                'traffic_density', 'traffic_speed'])        
        df_comb = track_df.merge(df_new, how='right', on='frame')
#        df_comb.to_csv(f'test/test_{l}.csv')

        df = pandas.concat([df, df_comb])

#    df.to_csv(f'test/all-test.csv')

    return df
        

def read_track_csv(arguments, df_stc, df_rec):
    """
    This method reads the tracks file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the tracks csv file.
    :return: a list containing all tracks as dictionaries.
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_path"])
    df = process_track_data(df, df_stc, df_rec)

    # Use groupby to aggregate track info. Less error prone than iterating over the data.
    grouped = df.groupby([TRACK_ID], sort=False)
    # Efficiently pre-allocate an empty list of sufficient size
    tracks = [None] * grouped.ngroups
    current_track = 0
    for group_id, rows in grouped:
#        bounding_boxes = np.transpose(np.array([rows[X].values,
#                                                rows[Y].values,
#                                                rows[WIDTH].values,
#                                                rows[HEIGHT].values]))
        tracks[current_track] = {TRACK_ID: np.int64(group_id),  # for compatibility, int would be more space efficient
                                 FRAME: rows[FRAME].values,
#                                 BBOX: bounding_boxes,
                                 X: rows[X].values,
                                 Y: rows[Y].values,
                                 X_VELOCITY: rows[X_VELOCITY].values,
                                 Y_VELOCITY: rows[Y_VELOCITY].values,
                                 X_ACCELERATION: rows[X_ACCELERATION].values,
                                 Y_ACCELERATION: rows[Y_ACCELERATION].values,
                                 FRONT_SIGHT_DISTANCE: rows[FRONT_SIGHT_DISTANCE].values,
                                 BACK_SIGHT_DISTANCE: rows[BACK_SIGHT_DISTANCE].values,
                                 THW: rows[THW].values,
                                 TTC: rows[TTC].values,
                                 DHW: rows[DHW].values,
                                 PRECEDING_X_VELOCITY: rows[PRECEDING_X_VELOCITY].values,
                                 PRECEDING_ID: rows[PRECEDING_ID].values,
                                 FOLLOWING_ID: rows[FOLLOWING_ID].values,
                                 LEFT_FOLLOWING_ID: rows[LEFT_FOLLOWING_ID].values,
                                 LEFT_ALONGSIDE_ID: rows[LEFT_ALONGSIDE_ID].values,
                                 LEFT_PRECEDING_ID: rows[LEFT_PRECEDING_ID].values,
                                 RIGHT_FOLLOWING_ID: rows[RIGHT_FOLLOWING_ID].values,
                                 RIGHT_ALONGSIDE_ID: rows[RIGHT_ALONGSIDE_ID].values,
                                 RIGHT_PRECEDING_ID: rows[RIGHT_PRECEDING_ID].values,
                                 LANE_ID: rows[LANE_ID].values,
                                 'Left_Pre_X': rows['Left_Pre_X'].values,
                                 'Left_Pre_Speed': rows['Left_Pre_Speed'].values,
                                 'Left_Al_X': rows['Left_Al_X'].values,
                                 'Left_Al_Speed': rows['Left_Al_Speed'].values,
                                 'Left_Fol_X': rows['Left_Fol_X'].values,
                                 'Left_Fol_Speed': rows['Left_Fol_Speed'].values,
                                 'Right_Pre_X': rows['Right_Pre_X'].values,
                                 'Right_Pre_Speed': rows['Right_Pre_Speed'].values,
                                 'Right_Al_X': rows['Right_Al_X'].values,
                                 'Right_Al_Speed': rows['Right_Al_Speed'].values,
                                 'Right_Fol_X': rows['Right_Fol_X'].values,
                                 'Right_Fol_Speed': rows['Right_Fol_Speed'].values,
                                 'traffic_density': rows['traffic_density'].values,
                                 'traffic_speed': rows['traffic_speed'].values,
                                 }
        current_track = current_track + 1
    return tracks


def read_static_info(arguments):
    """
    This method reads the static info file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_static_path"])
    
    def class_num(cat):
        if cat == 'Car':
            num = 0
        else:
            num = 1
        return num
    df[CLASS] = df[CLASS].apply(class_num)

    # Declare and initialize the static_dictionary
    static_dictionary = {}

    # Iterate over all rows of the csv because we need to create the bounding boxes for each row
    for i_row in range(df.shape[0]):
        track_id = int(df[TRACK_ID][i_row])
        static_dictionary[track_id] = {TRACK_ID: track_id,
                                       WIDTH: float(df[WIDTH][i_row]),
                                       HEIGHT: float(df[HEIGHT][i_row]),
                                       INITIAL_FRAME: int(df[INITIAL_FRAME][i_row]),
                                       FINAL_FRAME: int(df[FINAL_FRAME][i_row]),
                                       NUM_FRAMES: int(df[NUM_FRAMES][i_row]),
                                       CLASS: int(df[CLASS][i_row]),
                                       DRIVING_DIRECTION: float(df[DRIVING_DIRECTION][i_row]),
                                       TRAVELED_DISTANCE: float(df[TRAVELED_DISTANCE][i_row]),
                                       MIN_X_VELOCITY: float(df[MIN_X_VELOCITY][i_row]),
                                       MAX_X_VELOCITY: float(df[MAX_X_VELOCITY][i_row]),
                                       MEAN_X_VELOCITY: float(df[MEAN_X_VELOCITY][i_row]),
                                       MIN_TTC: float(df[MIN_TTC][i_row]),
                                       MIN_THW: float(df[MIN_THW][i_row]),
                                       MIN_DHW: float(df[MIN_DHW][i_row]),
                                       NUMBER_LANE_CHANGES: int(df[NUMBER_LANE_CHANGES][i_row])
                                       }
    return static_dictionary


def read_meta_info(arguments):
    """
    This method reads the video meta file from highD data.

    :param arguments: the parsed arguments for the program containing the input path for the video meta csv file.
    :return: the meta dictionary containing the general information of the video
    """
    # Read the csv file, convert it into a useful data structure
    df = pandas.read_csv(arguments["input_meta_path"])

    # Declare and initialize the extracted_meta_dictionary
    extracted_meta_dictionary = {ID: int(df[ID][0]),
                                 FRAME_RATE: int(df[FRAME_RATE][0]),
                                 LOCATION_ID: int(df[LOCATION_ID][0]),
                                 SPEED_LIMIT: float(df[SPEED_LIMIT][0]),
                                 MONTH: str(df[MONTH][0]),
                                 WEEKDAY: str(df[WEEKDAY][0]),
                                 START_TIME: str(df[START_TIME][0]),
                                 DURATION: float(df[DURATION][0]),
                                 TOTAL_DRIVEN_DISTANCE: float(df[TOTAL_DRIVEN_DISTANCE][0]),
                                 TOTAL_DRIVEN_TIME: float(df[TOTAL_DRIVEN_TIME][0]),
                                 N_VEHICLES: int(df[N_VEHICLES][0]),
                                 N_CARS: int(df[N_CARS][0]),
                                 N_TRUCKS: int(df[N_TRUCKS][0]),
                                 UPPER_LANE_MARKINGS: np.fromstring(df[UPPER_LANE_MARKINGS][0], sep=";"),
                                 LOWER_LANE_MARKINGS: np.fromstring(df[LOWER_LANE_MARKINGS][0], sep=";")}
    return extracted_meta_dictionary
