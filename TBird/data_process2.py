import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
from logparser import Spell, Drain
import argparse
from tqdm import tqdm
from logdeep.dataset.session import sliding_window
import time
import json

tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2

data_dir = os.path.expanduser("./")
output_dir = "../output/tbird/"
log_file = "Thunderbird_2M.log"

# construct other paths.
# data_dir = os.path.expanduser("~/.dataset/bgl_2k")
# output_dir = "/root/.dataset/output/bgl_2k/"
# log_file = "BGL_2k.log"
log_templates_file = os.path.join(output_dir, log_file + '_templates.csv')
log_structured_file = os.path.join(output_dir, log_file + '_structured.csv')
eventid_mapping_file = os.path.join(output_dir, f"{log_file}_log_templates.json")

# construct other paths.
train_file = os.path.join(output_dir, 'train')
test_normal_file = os.path.join(output_dir, 'test_normal')
test_abnormal_file = os.path.join(output_dir, 'test_abnormal')

# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f:
            total_size += 1
            if line.split(' ',1)[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))

def mapping():
    '''
    Function maps a unique integer to each EventId in descending order of 'Occurrences', 
    then writes this mapping to a .json.
    Input: log_templates_file - this should have been generated by parse_log().
    Output: .json with EventId mapped to a unique integer identifier.
    '''
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    print(f"\n\n ======= TODO: set MAX_TOKENS to {len(log_temp_dict)} in .env ======= \n\n")
    with open (eventid_mapping_file, "w") as f:
        json.dump(log_temp_dict, f)

# def deeplog_df_transfer(df, features, target, time_index, window_size):
#     """
#     :param window_size: offset datetime https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
#     :return:
#     """
#     agg_dict = {target:'max'}
#     for f in features:
#         agg_dict[f] = _custom_resampler
#
#     features.append(target)
#     features.append(time_index)
#     df = df[features]
#     deeplog_df = df.set_index(time_index).resample(window_size).agg(agg_dict).reset_index()
#     return deeplog_df
#
#
# def _custom_resampler(array_like):
#     return list(array_like)


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')

def get_longest_seq(filename):
    '''
    Function returns longest sequence in the file. 
    This is to be edited in .env. 
    This is important as BERT can only handle sequences of length 512 and below (including <CLS> and <SEP> tokens).
    '''
    with open(filename, "r") as f:
        counter = 0
        max_seq_len = 0
        sum_seq_len = 0
        exceeded_count = 0
        for line in f.readlines():
            counter += 1
            curr_seq_len = len(line.split())
            if curr_seq_len > 510:
                exceeded_count += 1
            sum_seq_len += curr_seq_len
            max_seq_len = max(len(line.split()), max_seq_len)

        print(f"|| ======== STATS FOR {filename} ======== ||")
        print(f"average seq length: {sum_seq_len / counter}")
        print(f"no. of sequences: {counter}")
        print(f"count of sequences with length > 510: {exceeded_count}({exceeded_count / counter * 100}%)")
        print(f"max_seq_length: {max_seq_len}")
      

def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+', #hexadecimal
        r'\d+.\d+.\d+.\d+',
        # r'/\w+( )$'
        r'\d+'
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=keep_para)
        parser.parse(log_file)

#
# def merge_list(time, activity):
#     time_activity = []
#     for i in range(len(activity)):
#         temp = []
#         assert len(time[i]) == len(activity[i])
#         for j in range(len(activity[i])):
#             temp.append(tuple([time[i][j], activity[i][j]]))
#         time_activity.append(np.array(temp))
#     return time_activity


if __name__ == "__main__":
    #
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', default=None, type=str, help="parser type")
    # parser.add_argument('-w', default='T', type=str, help='window size(mins)')
    # parser.add_argument('-s', default='1', type=str, help='step size(mins)')
    # parser.add_argument('-r', default=0.4, type=float, help="train ratio")
    # args = parser.parse_args()
    # print(args)
    #

    ##########
    # Parser #
    #########

    # print("parse log")
    # parse_log(data_dir, output_dir, log_file, 'drain')
    # print("mapping")
    # mapping() # creates mapping of EventId to a unique integer.
    # print("done mapping")

    #########
    # Count #
    #########
    # count_anomaly()

    ##################
    # Transformation #
    ##################
    # mins
    window_size = 5
    step_size = 1
    train_ratio = 0.4

    df = pd.read_csv(log_structured_file)

    # data preprocess
    df["Date"] = df["Date"].astype(str)
    df["Time"] = df["Time"].astype(str)
    df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)
    # convert time to UTC timestamp
    # df['deltaT'] = df['datetime'].apply(lambda t: (t - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

    # sampling with fixed window
    # features = ["EventId", "deltaT"]
    # target = "Label"
    # deeplog_df = deeplog_df_transfer(df, features, target, "datetime", window_size=args.w)
    # deeplog_df.dropna(subset=[target], inplace=True)

    # sampling with sliding window
    deeplog_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]],
                                para={"window_size": int(window_size), "step_size": int(step_size)}
                                )
    # deeplog_df = df[["timestamp", "Label", "EventId", "deltaT"]]
    print(deeplog_df.shape)
    print(deeplog_df["EventId"])
    
    # does the mapping of EventId to integer as determined by the eventid_mapping_file.
    with open(eventid_mapping_file, "r") as f:
        event_num = json.load(f)

    print(event_num)

    deeplog_df["EventId"] = deeplog_df["EventId"].apply(lambda x: [event_num.get(item, -1) for item in x])

    print(deeplog_df["EventId"])

    #########
    # Train #
    #########
    df_normal =deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    deeplog_file_generator(train_file, train, ["EventId"])

    print("training size {}".format(train_len))


    ###############
    # Test Normal #
    ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(test_normal_file, test_normal, ["EventId"])
    print("test normal size {}".format(normal_len - train_len))


    del df_normal
    del train
    del test_normal

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    #df_abnormal["EventId"] = df_abnormal["EventId"].progress_apply(lambda e: event_index_map[e] if event_index_map.get(e) else UNK)
    deeplog_file_generator(test_abnormal_file, df_abnormal, ["EventId"])
    print('test abnormal size {}'.format(len(df_abnormal)))
    
    # Get longest sequence to be set in .env.
    get_longest_seq(train_file)
    get_longest_seq(test_normal_file)
    get_longest_seq(test_abnormal_file)
    print(f"\n\n ======= TODO: set MAX_SEQ_LEN in .env to the max of the max_seq_length of the above 3 files ======= \n\n")
    time.sleep(50)