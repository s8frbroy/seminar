# import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def remove_motion_from_sensor(df, time_comp):

    '''
    remove redundant data due to refreshing cycles
    :param df: merged multi data int and float
    :return: df without refreshing cycles
    '''
    # list of entries which should be deleted
    delete = []

    sensor =  df.loc[1, "sensor_id"]

    if df.loc[0,"value"] == 0:
        n = 4
    else:
        n = 3

    for i in range(n, len(df), 2):

        cur = df.loc[i, "sensor_id"]
        prev = df.loc[i-1, "sensor_id"]

        ## print as update
        if i % 10001 == 0:
            print(i)

        assert df.loc[i,"sensor_id"] == sensor, "wrong sensor in df"
        assert df.loc[i, "value"] == df.loc[i-2, "value"], f"{i}"
        assert df.loc[i-1, "value"] == df.loc[i-3, "value"], f"{i}"

        time = df.loc[i-1, "time"] - df.loc[i-2, "time"]

        if time < time_comp and time >= 0:
            delete.append(i-1)
            delete.append(i-2)

        else:
            print(f'{i},{i-1}   diff: {time}')

    df = df.drop(delete)
    return df

def remove_error_refresh_cycle(df):

    '''
    remove redundant data due to refreshing cycles
    :param df: merged multi data int and float
    :return: df without refreshing cycles
    '''

    delete = []
    ## initialize dic:
    dic = {}
    for i in df['sensor_id'].value_counts().keys():
        dic[i] = -1

    for index, row in df.iterrows():

        sensor = row["sensor_id"]
        value = row["value"]

        if dic[sensor] == value:
            delete.append(index)

        dic[sensor] = value

        if index % 100000 == 0:
            print(index)

    df = df.drop(delete)
    return df


def create_on_off_Multi_char():

    '''
    dictionary for speed algorithm
    :return: Dictionary of char encoding
    '''

    dic_on_off = {}
    dic_on_off[5895] = ["a","A"]
    dic_on_off[7125] = ["b", "B"]
    dic_on_off[5896] = ["c", "C"]
    dic_on_off[6253] = ["d", "D"]
    dic_on_off[6632] = ["e", "E"]
    dic_on_off[6633] = ["f", "F"]
    dic_on_off[6635] = ["g", "G"]
    dic_on_off[6896] = ["h", "H"]
    dic_on_off[5887] = ["i", "I"]
    dic_on_off[5888] = ["j", "J"]
    dic_on_off[5889] = ["k", "K"]
    dic_on_off[5893] = ["l", "L"]
    return dic_on_off

def create_on_off_Multi_int():
    '''
    dictionary
    :return: Dictionary of int encoding
    '''
    dic_on_off = {}
    dic_on_off[5895] = [0,1]
    dic_on_off[7125] = [2, 3]
    dic_on_off[5896] = [4,5]
    dic_on_off[6253] = [6, 7]
    dic_on_off[6632] = [8, 9]
    dic_on_off[6633] = [10, 11]
    dic_on_off[6635] = [12, 13]
    dic_on_off[6896] = [14, 15]
    dic_on_off[5887] = [16, 17]
    dic_on_off[5888] = [18, 19]
    dic_on_off[5889] = [20, 21]
    dic_on_off[5893] = [22, 23]
    return dic_on_off

def add_event_col(df, dic):

    '''
    adds a column to the data set which combines value and event col
    :param df:
    :param dic: int or char representation
    :return:
    '''

    add = []

    for index, row in df.iterrows():

        sensor = row["sensor_id"]
        value = int(row["value"])

        add.append(dic[sensor][value])

    df["event"] = np.array(add)
    return df

def remove_error_motion(test):
    ## motion sensor data
    t_5895 = test[test["sensor_id"] == 5895]
    t_5893 = test[test["sensor_id"] == 5893]

    ## data without motion sensor
    test = test[test["sensor_id"] != 5895]
    test = test[test["sensor_id"] != 5893]

    ## reset index due to sampling
    test.reset_index(inplace=True)
    test.head()
    t_5895.reset_index(inplace=True)
    t_5893.reset_index(inplace=True)

    ## set time const to somehow deal with unbalanced data set
    time_comp_5895 = 50
    time_comp_5893 = 15

    ## remove redundant data
    t_5895_t = remove_motion_from_sensor(t_5895, time_comp_5895)
    t_5893_t = remove_motion_from_sensor(t_5893, time_comp_5893)

    test = test[["sensor_id", "value", "time", "timestamp"]]
    t_5895_t = t_5895_t[["sensor_id", "value", "time", "timestamp"]]
    t_5893_t = t_5893_t[["sensor_id", "value", "time", "timestamp"]]

    ## merge datasets back together
    merge = merge_multi_data(test, t_5895_t)
    merge = merge_multi_data(merge, t_5893_t)

    return merge


def load_multi_data(filename):
    '''
    :param number_upper: upper file number
    :param number_lower: lower file number
    :param type: "int" or "float"
    :return: dataframe
    '''


    ## sensor_ids to keep
    ids = [5895,7125,5896,6253,5887,5888,5889,5893,6632,6633,6635,6896]
    dic = pd.read_csv(filename, names=['value_id', 'sensor_id', 'timestamp', 'value'])
    dic = dic[dic['sensor_id'].isin(ids)]
    dic = prepare_time_multi(dic)
    return dic


def prepare_time_multi(multi):
    '''
    return: multi data set with time as [0,86400] format
    '''
    hour = pd.DatetimeIndex(multi["timestamp"]).hour
    minute = pd.DatetimeIndex(multi["timestamp"]).minute
    second = pd.DatetimeIndex(multi["timestamp"]).second
    multi['time'] = second + 60 * minute + 3600 * hour

    return multi

def merge_multi_data(multi_int, multi_float):
    '''
    :param multi_int : preprocessed multi_int df
    :param multi_float : preprocessed multi_float df
    :return: merged_data "[sorted data]"
    '''
    df = pd.concat([multi_int, multi_float], ignore_index=True, sort=False)
    merged_data = df.sort_values(by=['timestamp'])
    return merged_data

def merge_aras_data(multi_int, multi_float):
    '''
    :goal:   helper function for removing errors
    :return: merged_data "[sorted data]"
    '''
    df = pd.concat([multi_int, multi_float], ignore_index=True, sort=False)
    merged_data = df.sort_values(by=['Unnamed: 0'])
    return merged_data

def load_aras(filename, house):
    '''
    :param house: "HouseA" or "HouseB"
    :param days:  how many days you want to load, 1 day = 86400 lines
    :return: dataframe
    '''
    names = [str(x) for x in range(0, 22)]
    dic = pd.read_csv(filename, sep=" ", names=names)
    dic['time'] = list(range(1, 86401))
    dic = remove_sensors_not_used(dic, house)
    dic = remove_rows_Aras(dic)
    return dic


def remove_sensors_not_used(df, house):
    '''
    :param df: dataframe
    :param house: specify "HouseA" or "HouseB"
    :return: dataframe only with the usable sensor
    '''

    if house == "HouseA":
        return remove_sensors_HouseA(df)
    else:
        if house == "HouseB":
            return remove_sensors_HouseB(df)
        else:
            raise ValueError


def remove_sensors_HouseA(df):
    '''
    :param df: df
    :return: df with sensor: 4,5,8,12,13,14,16,17,18,19,20
    '''
    sensors_keep = np.array(["3", "4", "7", "11", "12", "13", "15", "16", "17", "19", "20", "21", "time"])
    return df.loc[:, sensors_keep]


def remove_sensors_HouseB(df):
    '''
    :param df: df
    :return: df with sensor: 3,4,5,9,11,12,13,14,15,16,17,18,19,20
    '''
    sensors_keep = np.array(["2", "5", "6", "10", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "time"])
    return df.loc[:, sensors_keep]

def remove_rows_Aras(df):
    '''
    :return: df without duplicate rows
    '''
    df = df.reset_index(drop=True)
    cur = df.iloc[0, :]
    i = 0
    delete = []
    for index, row in df.iterrows():
        
        ## Just for Updating reason
        if i % 20000 == 0:
            print(i)

        assert i == index, f"{i}, index:{index}"
        i += 1
        if index == 0:
            pass

        else:
            ## get previous and current column
            pre = cur
            cur = row

            ## compare their sensor values
            comp = cur[:-3] == pre[:-3]
            ## delete current one if all sensor values are the same
            if comp.all():
                delete.append(index)


    df = df.drop(delete)
    return df


def transform_House(df):
    '''
    :goal: new column to identity which sensor changed to which value
    '''
    df = df.reset_index(drop=True)
    sensor = []
    value = []

    for i in range(1, len(df)):
        last = df.iloc[i-1, :-3]
        cur = df.iloc[i,:-3]

        comp = last == cur
        assert comp.all() != True

        ind = int(comp[comp == False].index[0])
        sensor.append(ind)
        value.append(df.loc[i, str(ind)])

    df = df.iloc[1:,]
    df["sensor"] = sensor
    df["value"] = value

    return df

def convert_to_multi(data, type):
    if type == "HouseA":
        return convert_to_multi_sensor_H_A(data)
    else:
        return convert_to_multi_sensor_H_B(data)

def convert_to_multi_sensor_H_A(df):
    '''
    :goal: transform Aras HouseA data into Multi sensor like form
    :df:   data frame with sensor_id column of corresponding Multi data sensor 
    '''
    df = df.reset_index(drop = True)

    time = []
    sensor_id = []
    value = []
    old = df.loc[:, "sensor"]
    print(len(old))
    for i in range(0, len(df)):

        sen = int(df.loc[i, "sensor"])
        tim = df.loc[i,"time"]
        val = df.loc[i,"value"]

        list = [12,13,17,16,19,7,15,11,3,4]
        ## check for missing sensor
        if sen not in list:
            print(i)
            print(type(sen))
            print(sen)
            raise ValueError

        elif sen == 12:
            time.append(tim)
            sensor_id.append(5895)
            value.append(val)

        elif sen == 13:
            time.append(tim)
            sensor_id.append(7125)
            value.append(val)

        elif sen == 17:
            time.append(tim)
            sensor_id.append(7125)
            value.append(val)

        elif sen == 16:
            time.append(tim)
            sensor_id.append(7125)
            value.append(val)

        elif sen == 19:
            time.append(tim)
            sensor_id.append(5896)
            value.append(val)

        elif sen == 7:
            time.append(tim)
            sensor_id.append(6253)
            value.append(val)

        elif sen == 15:
            time.append(tim)
            sensor_id.append(5893)
            value.append(val)

        elif sen == 11:
            time.append(tim)
            sensor_id.append(5888)
            value.append(val)

        elif sen == 3:
            time.append(tim)
            sensor_id.append(5889)
            value.append(val)

        elif sen == 4:
            time.append(tim)
            sensor_id.append(5889)
            value.append(val)
        else:
            print(f"Missing: {sen}")


    dt = {'sensor_id' : sensor_id, 'value' : value, "time" : time, "old_sen" : old}
    data = pd.DataFrame(data = dt)

    return data

def one_hot_encoding(df):

    df = df.reset_index(drop = True)
    ## initialize 24 * n_datapoints array
    len = df.shape[0]
    n_event = 24
    enc = [[0 for i in range(len)] for j in range(n_event)]
    enc = np.array(enc)

    ## fill array
    for i in range(0, df.shape[0]):

        sen = df.loc[i, "event"]
        enc[sen, i] = 1

    dt = {"time_scal" :df["time_scal"],'0' : enc[0], '1': enc[1], '2' : enc[2], '3': enc[3], '4': enc[4], '5': enc[5],
          '6' : enc[6], '7': enc[7], '8' : enc[8], '9': enc[9], '10': enc[10], '11': enc[11],
          '12' : enc[12], '13': enc[13], '14' : enc[14], '15': enc[15], '16': enc[16], '17': enc[17],
          '18' : enc[18], '19': enc[19], '20' : enc[20], '21': enc[21], '22': enc[22], '23': enc[23]
          }
    data = pd.DataFrame(data = dt)
    print(data.shape)
    return data


def convert_to_multi_sensor_H_B(df):
    '''
    :goal: transform Aras HouseB data into Multi sensor like form
    '''
    df = df.reset_index(drop = True)

    time = []
    sensor_id = []
    value = []

    old = df.loc[:, "sensor"]
    print(len(old))

    for i in range(0, len(df)):

        sen = df.loc[i, "sensor"]
        tim = df.loc[i,"time"]
        val = df.loc[i,"value"]

        if sen == 17:
            time.append(tim)
            sensor_id.append(5895)
            value.append(val)

        elif sen == 6 :
            time.append(tim)
            sensor_id.append(7125)
            value.append(val)
        elif sen == 5:
            time.append(tim)
            sensor_id.append(7125)
            value.append(val)

        elif sen == 19:
            time.append(tim)
            sensor_id.append(7125)
            value.append(val)

        elif sen == 14 :
            time.append(tim)
            sensor_id.append(5896)
            value.append(val)

        elif sen == 15:
            time.append(tim)
            sensor_id.append(5896)
            value.append(val)

        elif sen == 10:
            time.append(tim)
            sensor_id.append(6253)
            value.append(val)

        elif sen == 18:
            time.append(tim)
            sensor_id.append(5893)
            value.append(val)

        elif sen == 2:
            time.append(tim)
            sensor_id.append(5888)
            value.append(val)

        elif sen == 12:
            time.append(tim)
            sensor_id.append(5889)
            value.append(val)
        elif sen == 13:
            time.append(tim)
            sensor_id.append(5889)
            value.append(val)
        elif sen == 16:
            time.append(tim)
            sensor_id.append(5889)
            value.append(val)

        else:
            print(sen)
            print (i)
            raise ValueError

    dt = {'sensor_id' : sensor_id, 'value' : value, "time" : time, "old_sen" : old}
    data = pd.DataFrame(data = dt)

    return data



def convert_to_binary(df):

    '''
    :Goal: Convert Multi Sensor float and int values to Binary
    '''

    ## convert stove light 5887 (<600 = 1)
    df.loc[((df['sensor_id'] == 5887) & (df['value'] <=  600)), ['value']] = 0
    df.loc[((df['sensor_id'] == 5887) & (df['value'] >  600)), ['value']] = 1

    ## convert couch pressure 5889 (<285 = 0)
    df.loc[((df['sensor_id'] == 5889) & (df['value'] <=  285)), ['value']] = 0
    df.loc[((df['sensor_id'] == 5889) & (df['value'] >  285)), ['value']] = 1

    ## convert bed pressure 5896 (<630 = 0)
    df.loc[((df['sensor_id'] == 5896) & (df['value'] <=  630)), ['value']] = 0
    df.loc[((df['sensor_id'] == 5896) & (df['value'] >  630)), ['value']] = 1

    ## convert coffemaker 6632 (<100 = 0)
    df.loc[((df['sensor_id'] == 6632) & (df['value'] <=  100)), ['value']] = 0
    df.loc[((df['sensor_id'] == 6632) & (df['value'] >  100)), ['value']] = 1

    ## convert sandwich maker 6633 (<500 = 0)
    df.loc[((df['sensor_id'] == 6633) & (df['value'] <=  500)), ['value']] = 0
    df.loc[((df['sensor_id'] == 6633) & (df['value'] >  500)), ['value']] = 1

    ## convert kettle 6635 (<100)
    df.loc[((df['sensor_id'] == 6635) & (df['value'] <=  100)), ['value']] = 0
    df.loc[((df['sensor_id'] == 6635) & (df['value'] >  100)), ['value']] = 1

    ## convert microwave 6896 (<500 = 0 )
    df.loc[((df['sensor_id'] == 6896) & (df['value'] <=  500)), ['value']] = 0
    df.loc[((df['sensor_id'] == 6896) & (df['value'] >  500)), ['value']] = 1

    ## convert bathroom light (<90 = 0 )
    df.loc[((df['sensor_id'] == 7125) & (df['value'] <=  90)), ['value']] = 0
    df.loc[((df['sensor_id'] == 7125) & (df['value'] >  90)), ['value']] = 1

    ############################################################################
    ###############################   useless   ################################
    ############################################################################

    ## convert entrance contact 5888 (==1 = 1)
    df.loc[((df['sensor_id'] == 5888) & (df['value'] <  1)), ['value']] = 0
    df.loc[((df['sensor_id'] == 5888) & (df['value'] >= 630)), ['value']] = 1

    return df