import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import math, time, random
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import spearmanr
from scipy.optimize import linprog
import io
from collections import defaultdict
import pulp
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
import argparse

from model import *


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Pipeline:
1. Aggregate and preprocess data
2. GMS predictions (a list of pandas df's)
3. Calculate seasonality score and filter with it.
4. Postprocess (zero-out non-increasing discount predictions & remove constant items)
5. Optimization through linear programming
'''

args = argparse.ArgumentParser()
args.add_argument('--TIME_SCALE', nargs='?', type=str, default='month') # month or week
args.add_argument('--TAXONOMY_LEVEL', nargs='?', type=str, default='catg') # catg or subcatg
args.add_argument('--PATH_DATA_RAW', nargs='?', type=str, default='./') # raw data path
args.add_argument('--PATH_SAVE', nargs='?', type=str, default='./') # save path
args.add_argument('--THRESHOLD_GMS_OUTLIER', nargs='?', type=float, default=1e8) # outlier threshold
args.add_argument('--THRESHOLD_SEASONALITY', nargs='?', type=float, default=0.3)
args.add_argument('--BUDGET', nargs='?', type=float, default=1e6)
args.add_argument('--DATA_AGGREGATED', nargs='?', type=bool, default=False) # whether data is algready aggregated (saves time when testing)
args.add_argument('--DATA_CREATED', nargs='?', type=bool, default=False)
args.add_argument('--CALIBRATION_MODE', nargs='?', type=str, default='item') # calibration methods: none, item, time, all
args.add_argument('--DEBUG', nargs='?', type=bool, default=False)
# LSTM params
args.add_argument('--lr', nargs='?', type=float, default=0.00001) # learing rate
args.add_argument('--model_type', nargs='?', type=str, default='LSTM') # LSTM, GRU, LSTMCONV
args.add_argument('--TEST_SET_SIZE', nargs='?', type=float, default=0.2)
args.add_argument('--CLIP_GRAD', nargs='?', type=float, default=100.)
args.add_argument('--BATCH_SIZE', nargs='?', type=int, default=128)

args.add_argument('--HIDDEN_DIM', nargs='?', type=int, default=32)
args.add_argument('--LATENT_DIM', nargs='?', type=int, default=1024)
args.add_argument('--N_LAYERS', nargs='?', type=int, default=4)
args.add_argument('--EPOCHS', nargs='?', type=int, default=200)
args.add_argument('--BI_DIRECTIONAL', nargs='?', type=bool, default=False)
# CONV params
args.add_argument('--NUM_KERNELS1', nargs='?', type=int, default=20) # how many kernels per channel for conv1
args.add_argument('--NUM_KERNELS2', nargs='?', type=int, default=5) # how many kernels per channel for conv2
args = args.parse_args()

args.PATH_MODEL_SAVE = args.PATH_SAVE + 'model_{}_{}.pt'.format(args.TAXONOMY_LEVEL, args.TIME_SCALE)
args.CONTEXT_SIZE = 24 if args.TIME_SCALE == 'month' else 96 # 2 years
args.FORECAST_SIZE = 12 if args.TIME_SCALE == 'month' else 48 # 1 year






args.PATH_DATA_RAW = 'https://raw.githubusercontent.com/dykim1222/gmsdata/master/catg_mnth.csv'
df = pd.read_csv(args.PATH_DATA_RAW)

predictor = Predictor(df, args)                     # initialization
predictor.aggregate()                               # data aggregation
predictor.preprocess()                              # data preprocess
predictor.generate_dataset()                        # data generation
predictor.train()                                   # train and save the model
gms_csv = predictor.infer()                         # inference
gms_csv = predictor.apply_season_filter(gms_csv)    # filtering with seasonality
gms_csv = predictor.postprocess(gms_csv)            # POSTPROCESS
promo_cal = predictor.optimize(gms_csv)             # OPTIMIZATION
