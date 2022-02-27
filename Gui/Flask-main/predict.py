import numpy as np
import tensorflow as tf
import os
import sys
import base64
import json

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
    
def init_model_speed(dataset):

    # loads statistics for each dataset from a dictionay, and builds the nested list as tree
    if dataset == "HouseA":
        dict_tree_filename = "ARAS_House_A_dict.json"
    elif dataset == "HouseB":
        dict_tree_filename = "ARAS_House_B_dict.json"
    elif dataset == "Multi":
        dict_tree_filename = "Multisensor_dict.json"

    loaded_model = build_tree(dict_tree_filename, dataset)
     
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


# Processing for SPEED
def preprocessing_speed(df_raw):
    dic_on_off = create_on_off_Multi_char()

    df_clean = df_raw
    df = add_event_col(df_clean, dic_on_off)
  
    new_df = df[["time", "event"]]
    processed_data = new_df
    
    processed_list = processed_data.values.tolist()
    
    sequence_list = ""
    for item in processed_list:
        sequence_list += item[1]

    return processed_data, sequence_list

def add_event_col(df, dic):

    add = []

    for index, row in df.iterrows():

        sensor = row["sensor_id"]
        value = int(row["value"])

        add.append(dic[sensor][value])

    df["event"] = np.array(add)
    return df

def create_on_off_Multi_char():
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
    

# make a prediction based on image
def predict(model,dataX):
    prediction = model.predict(dataX)
    output = prediction
    return output
    
    
# processing for SPEED
def build_tree(dict_tree_filename, dataset):
    #dict_tree = create_dict_for_tree(episode_list, dict_tree)
    
    with open(dict_tree_filename) as handle:
        dict_tree = json.loads(handle.read())

    the_tree, count_of_all_nodes = build(dict_tree, dataset)
    
    return the_tree, count_of_all_nodes

    
def build(dict_tree, dataset):
    the_tree = list()
    count_of_all_nodes = dict()
 
    for key, item in dict_tree.items(): # key: root node (always caps eg. ABC, one of the 12)
        key_list = item[1].keys()
        sub_tree = list()
        checking = []
        check_now = ""
        count_nodes(count_of_all_nodes, key)
        for sub_key in key_list:  # sub_key: the first alphabet following the root (could be upper/lower case)
            if len(checking) == 0 or len(sub_key) == 1:   # adding the first node after the root
                  checking.append(sub_key)    # add into the list of existing nodes (for later checking)
                  check_now = sub_key
                  sub_tree.append([check_now, item[1][sub_key]])    # adds to subtree
                  count_nodes(count_of_all_nodes, item[1][sub_key])
            elif sub_key.startswith(checking[-1]):        # match to the last node, if True the next node should be a child of the last one
                append_node_char = sub_key[len(checking[-1]):]
                if len(append_node_char) == 1:
                  sub_tree = tree_search(sub_tree, check_now, append_node_char, item[1][sub_key])
                  count_nodes(count_of_all_nodes, append_node_char)
                check_now = sub_key
                checking.append(check_now)
            elif sub_key.startswith(checking[-1]) == False:
                re_check_id = -2  #check one more index back
                check_now = checking[re_check_id]
                while check_now != checking[0]:
                    if sub_key.startswith(checking[re_check_id]):
                        append_node_char = sub_key[len(checking[re_check_id]):]
                        sub_tree = tree_search(sub_tree, check_now, append_node_char, item[1][sub_key])
                        count_nodes(count_of_all_nodes, append_node_char)
                        check_now = sub_key
                        checking.append(check_now)
                        break
                    else:
                        re_check_id -= 1
                        check_now = checking[re_check_id]
        the_tree.append([key, count_of_all_nodes[key], sub_tree])

    return the_tree, count_of_all_nodes

def count_nodes(node_dict, node):
    caps = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]   # turning on
    lows = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]   # turning off

    all_actions = caps + lows
    if node in caps or node in lows:
        if node in node_dict:
            node_dict[node] += 1
        else:
            node_dict[node] = 1
    return node_dict
    
def speed_predict(model, input, count_of_all_nodes):
    stat_tree = model
    output = speed_predict_from_tree(model, input, count_of_all_nodes)
    return output
    
    
def speed_predict_from_tree(the_tree, check_seq, count_of_all_nodes):
    predict_prob = list()
    total_count_of_nodes = 0
    for key, item in count_of_all_nodes.items():
        total_count_of_nodes += item
    
    for act in all_actions:
        target = act
        if target in count_of_all_nodes.keys():
          the_prob = prob_of_target(check_seq, target, the_tree, total_count_of_nodes, count_of_all_nodes)
          predict_prob.append([act, the_prob])
        else:
          predict_prob.append([act, 0])

    prediction = max(predict_prob, key=lambda x:x[1])
    return prediction, predict_prob
    
def prob_of_target(check_seq, target, the_tree, total_count_of_nodes, count_of_all_nodes):
    root = check_seq[0]
    root_subtree = list()
    store_for_prob = list()
    target_count_in_term_node = 0

    store_for_prob, target_count_in_term_node, root_subtree = stored_for_prob(target_count_in_term_node, check_seq, target, the_tree)

    target_in_all_nodes = count_of_all_nodes[target]/total_count_of_nodes
    prob_output = prob(root, target, check_seq, store_for_prob, target_count_in_term_node, root_subtree, count_of_all_nodes, total_count_of_nodes)

    return prob_output
    
def prob(root, target, check_seq, store_for_prob, target_count_in_term_node, root_subtree, count_of_all_nodes, total_count_of_nodes):
    target_in_all_nodes = count_of_all_nodes[target]/total_count_of_nodes
    prob = (target_count_in_term_node/root_subtree[1])*target_in_all_nodes
    prob_output = prob

    for item in store_for_prob:
        new_prob = (target_count_in_term_node/item[1])*prob_output
        prob_output = new_prob
    return prob_output


def stored_for_prob(target_count_in_term_node, check_seq, target, the_tree):
    store_for_prob = list()
    for root in the_tree:
        if root[0] == check_seq[0]:
            root_subtree = root
            check_subseq = check_seq[1:]
            counter = len(check_subseq)
            check_layer = root[2]

            while counter > -2:
                for sub_item in check_layer:
                    store_for_prob, target_count_in_term_node = get_store_list(target_count_in_term_node, check_seq, target, the_tree, check_subseq, sub_item, store_for_prob, counter)
                    check_subseq = check_subseq[1:]
                counter -= 1
    return (store_for_prob, target_count_in_term_node, root_subtree)