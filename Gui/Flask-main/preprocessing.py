import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util

def process_data_Aras(filename, type):
    return preprocessing_Aras(filename, type)
    
def process_data_Multi(filename_float, filename_int):
    return preprocess_multi(filename_float, filename_int)

def preprocessing_Aras(filename, type):

    ## preprocess HouseA or HouseB
    assert (type == "HouseA") or (type == "HouseB")
    data = util.load_aras(filename, type)
    data = util.transform_House(data)
    data = util.convert_to_multi(data, type)
    return data

def preprocess_multi(filename_float, filename_int):

    ## load int and float data
    multi_float = util.load_multi_data(filename_float)
    multi_int   = util.load_multi_data(filename_int)

    ## merge float and int dataset
    df = util.merge_multi_data(multi_int, multi_float)

    ## convert float and int values to binary
    binary = util.convert_to_binary(df)

    df = binary[["sensor_id", "value", "time", "timestamp"]]

    ## index is wrong due to merging the data sets
    df.reset_index(drop=True)

    ## remove erros due to refreshing cycle
    df = util.remove_error_refresh_cycle(df)

    ## remove errors due to motion sensor
    df = util.remove_error_motion(df)

    return df
