import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import torch
import random
import csv

import os

thresholds = []

# folder path.
folder_dir = input("Please input the folder path containing 'normal_anomalies.csv' and 'abnormal_anomalies.csv'.")
threshold = float(input("Please input a threshold."))


# construct paths.
normal_anomalies_file = os.path.join(folder_dir, "expt4_harmonic_mean_normal_anomalies.csv")
abnormal_anomalies_file = os.path.join(folder_dir, "expt4_harmonic_mean_abnormal_anomalies.csv")
# best_threshold

# for threshold in thresholds:

with open(normal_anomalies_file, 'r') as normal_anomalies_file:
    normal_anomaly_scores = normal_anomalies_file.read().split(',')
    normal_anomaly_scores = [float(value) for value in normal_anomaly_scores]

with open(abnormal_anomalies_file, 'r') as abnormal_anomalies_file:
    abnormal_anomaly_scores = abnormal_anomalies_file.read().split(',')
    abnormal_anomaly_scores = [float(value) for value in abnormal_anomaly_scores]



normal_positives = sum(1 for value in normal_anomaly_scores if value > threshold)
abnormal_positives = sum(1 for value in abnormal_anomaly_scores if value > threshold)
normal_negatives = sum(1 for value in normal_anomaly_scores if value <= threshold)
abnormal_negatives = sum(1 for value in abnormal_anomaly_scores if value <= threshold)
precision = abnormal_positives / (abnormal_positives + normal_positives)
recall = abnormal_positives / (abnormal_positives + abnormal_negatives)
f1 = (2 * precision * recall) / (precision + recall)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1} using threshold {threshold}")


