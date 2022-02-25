import numpy as np
import tensorflow as tf
import os
import sys
import base64

from main import *
from preprocessing import *
from util import *

sys.path.append(os.path.abspath("./model"))


# initialize model
def init_model_cnn():
    loaded_model = tf.keras.models.load_model('model/cnn_model.h5')
    print("loaded model successfully")
    return loaded_model

def init_model_lstm():
    loaded_model = tf.keras.models.load_model('model/lstm_model.h5')
    print("loaded model successfully")
    return loaded_model

#Processing for NN models
def preprocessing_2(df):
    dic = util.create_on_off_Multi_int()
    df = util.add_event_col(df, dic)
    df["time_scal"] = df["time"]/ 86400
    df = df[["event", "time_scal"]]
    df.to_csv("static/files/data.csv")

    global X_test
    global y_test
    df2 = util.one_hot_encoding(df)
    # dummies2 = pd.get_dummies(df.event)
    # df2 = pd.concat([df,dummies2], axis ='columns')
    #df2.drop(['event'],axis=1,inplace=True)
    X_test =[]
    y_test =[]
    test_data = df2.values
    for i in range(20,len(test_data)):
        X_test.append(test_data[i-20:i, :])
        y_test.append(test_data[i, 1:])
    X_test, y_test = np.array(X_test), np.array(y_test)
    return X_test

#choose model

# make a prediction based on image
def predict(model,dataX):
    prediction = model.predict(dataX)
    output = prediction
    return output

