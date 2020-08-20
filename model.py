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

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_printoptions(sci_mode=False)

# DEEP LEANING MODELS
class LSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, args, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.args = args
        if self.args.N_LAYERS == 1:
            drop_prob = 0.
        self.n_dirs = 2 if args.BI_DIRECTIONAL else 1

        self.embeddings = nn.Embedding(self.args.VOCAB_SIZE, self.args.EMBEDDING_DIM)
        self.lstm = nn.LSTM(input_dim, self.args.HIDDEN_DIM, self.args.N_LAYERS, batch_first=True, dropout=drop_prob, bidirectional = self.args.BI_DIRECTIONAL)
        self.bn1 = nn.BatchNorm1d((self.args.CONTEXT_SIZE + 2*self.args.N_LAYERS)*self.n_dirs*self.args.HIDDEN_DIM)
        self.fc1 = nn.Linear((self.args.CONTEXT_SIZE + 2*self.args.N_LAYERS)*self.n_dirs*self.args.HIDDEN_DIM, self.args.LATENT_DIM)
        self.bn2 = nn.BatchNorm1d(self.args.LATENT_DIM)
        self.fc2 = nn.Linear(self.args.LATENT_DIM, self.args.LATENT_DIM)

        for i in range(self.args.FORECAST_SIZE):
            self.add_module('head{}'.format(i), \
                            nn.Sequential(
                                nn.BatchNorm1d(self.args.LATENT_DIM + 1),
                                nn.Linear(self.args.LATENT_DIM + 1, self.args.LATENT_DIM),
                                nn.ReLU(),
                                nn.BatchNorm1d(self.args.LATENT_DIM),
                                nn.Linear(self.args.LATENT_DIM, output_dim)
                                )  )

    def forward(self, x, dsc, h):
        out = torch.cat((self.embeddings(x[:,:,0].long()), x[:,:,1:]), dim=2)
        out, h = self.lstm(out, h)
        out = F.relu(out.reshape(out.shape[0], -1))
        h, c = h[0].transpose(0,1).reshape(out.shape[0],-1), h[1].transpose(0,1).reshape(out.shape[0],-1)
        out = torch.cat((out, h, c), dim=1)

        out = F.relu(self.fc2(self.bn2(F.relu(self.fc1(self.bn1(out))))))
        out = torch.cat([self._modules['head{}'.format(i)](torch.cat((out, dsc[:,i].reshape(-1,1)), dim=1)) for i in range(self.args.FORECAST_SIZE)], dim=1)
        return out.sigmoid()

    def init_hidden(self, bs):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_dirs*self.args.N_LAYERS, bs, self.args.HIDDEN_DIM).zero_().to(device),
                  weight.new(self.n_dirs*self.args.N_LAYERS, bs, self.args.HIDDEN_DIM).zero_().to(device))
        return hidden

class LSTMConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, args, drop_prob=0.2):
        super(LSTMConvNet, self).__init__()
        self.args = args



        if self.args.N_LAYERS == 1:
            drop_prob = 0.
        self.n_dirs = 2 if args.BI_DIRECTIONAL else 1

        # Embedding layer
        self.embeddings = nn.Embedding(args.VOCAB_SIZE, args.EMBEDDING_DIM)

        # Convolution layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=args.NUM_KERNELS1*input_dim, padding=args.CONTEXT_SIZE-1, kernel_size=args.CONTEXT_SIZE, groups=input_dim)
        self.conv2 = nn.Conv1d(in_channels=self.conv1.out_channels, out_channels=args.NUM_KERNELS2*input_dim, kernel_size=args.CONTEXT_SIZE, groups=input_dim)
        self.conv_cat_size = self.conv2.out_channels * args.CONTEXT_SIZE

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, args.HIDDEN_DIM, args.N_LAYERS, batch_first=True, dropout=drop_prob, bidirectional = args.BI_DIRECTIONAL)
        self.lstm_cat_size = (args.CONTEXT_SIZE + 2*args.N_LAYERS)*self.n_dirs*args.HIDDEN_DIM

        self.all_cat_size = args.CONTEXT_SIZE*input_dim + self.conv_cat_size + self.lstm_cat_size
        self.bn1 = nn.BatchNorm1d(self.all_cat_size)
        self.fc1 = nn.Linear(self.all_cat_size, 10*args.LATENT_DIM)
        self.bn2 = nn.BatchNorm1d(10*args.LATENT_DIM)
        self.fc2 = nn.Linear(10*args.LATENT_DIM, args.LATENT_DIM)

        # Multi-head layer
        for i in range(args.FORECAST_SIZE):
            self.add_module('head{}'.format(i), \
                            nn.Sequential(
                                nn.BatchNorm1d(args.LATENT_DIM + 1),
                                nn.Linear(args.LATENT_DIM + 1, args.LATENT_DIM),
                                nn.ReLU(),
                                nn.BatchNorm1d(args.LATENT_DIM),
                                nn.Linear(args.LATENT_DIM, output_dim)
                                )  )

    def forward(self, x, dsc, h):
        # embedding
        out = torch.cat((self.embeddings(x[:,:,0].long()), x[:,:,1:]), dim=2)

        # convolution
        out_conv = self.conv2(F.relu(self.conv1(out.transpose(1,2)))).reshape(out.shape[0], -1)

        # lstm
        out_lstm, hc = self.lstm(out, h)
        out_lstm = out_lstm.reshape(out.shape[0], -1)
        h, c = hc[0].transpose(0,1).reshape(out.shape[0],-1), hc[1].transpose(0,1).reshape(out.shape[0],-1)
        out_lstm = torch.cat((out_lstm, h, c),dim=1)

        # concatenation and dense layers
        out = torch.cat((out.reshape(out.shape[0], -1), out_conv, out_lstm), dim=1)
        out = F.relu(self.fc2(self.bn2(F.relu(self.fc1(self.bn1(F.relu(out)))))))

        # multi-heads
        out = torch.cat([self._modules['head{}'.format(i)](torch.cat((out, dsc[:,i].reshape(-1,1)), dim=1)) for i in range(self.args.FORECAST_SIZE)], dim=1)
        return out.sigmoid()

    def init_hidden(self, bs):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_dirs*self.args.N_LAYERS, bs, self.args.HIDDEN_DIM).zero_().to(device),
                  weight.new(self.n_dirs*self.args.N_LAYERS, bs, self.args.HIDDEN_DIM).zero_().to(device))
        return hidden

class GRUNet(nn.Module):
    def __init__(self, input_dim, output_dim, args, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.args = args

        if args.N_LAYERS == 1:
            drop_prob = 0.
        self.n_dirs = 2 if args.BI_DIRECTIONAL else 1

        self.embeddings = nn.Embedding(args.VOCAB_SIZE, args.EMBEDDING_DIM)
        self.gru = nn.GRU(input_dim, args.HIDDEN_DIM, args.N_LAYERS, batch_first=True, dropout=drop_prob, bidirectional=args.BI_DIRECTIONAL)
        self.bn1 = nn.BatchNorm1d(self.n_dirs*(args.CONTEXT_SIZE + args.N_LAYERS)*args.HIDDEN_DIM)
        self.fc1 = nn.Linear((args.CONTEXT_SIZE + args.N_LAYERS)*self.n_dirs*args.HIDDEN_DIM, args.LATENT_DIM)
        self.bn2 = nn.BatchNorm1d(args.LATENT_DIM)
        self.fc2 = nn.Linear(args.LATENT_DIM, args.LATENT_DIM)

        for i in range(args.FORECAST_SIZE):
            self.add_module('head{}'.format(i), \
                            nn.Sequential(
                                nn.BatchNorm1d(args.LATENT_DIM + 1),
                                nn.Linear(args.LATENT_DIM + 1, args.LATENT_DIM),
                                nn.ReLU(),
                                nn.BatchNorm1d(args.LATENT_DIM),
                                nn.Linear(args.LATENT_DIM, output_dim)
                                )  )

    def forward(self, x, dsc, h):
        out = torch.cat((self.embeddings(x[:,:,0].long()), x[:,:,1:]), dim=2)
        out, h = self.gru(out, h)
        out = F.relu(out.reshape(out.shape[0], -1))
        h = h.transpose(0,1).reshape(out.shape[0],-1)
        out = torch.cat((out,h), dim=1)

        out = F.relu(self.fc2(self.bn2(F.relu(self.fc1(self.bn1(out))))))
        out = torch.cat([self._modules['head{}'.format(i)](torch.cat((out, dsc[:,i].reshape(-1,1)), dim=1)) for i in range(self.args.FORECAST_SIZE)], dim=1)
        return out.sigmoid() #logits

    def init_hidden(self, bs):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_dirs*self.args.N_LAYERS, bs, self.args.HIDDEN_DIM).zero_().to(device)
        return hidden

# SEASONALITY SCORE
class Seasonalizer:
    def __init__(self, dp, args):
        # dp.catg_id is catg index right now
        # gms_tensor is catg index right now
        self.dp = dp.copy()
        self.args = args

    def calculate(self):
        # calculate_seasonality
        def calculate_seasonality(ds):
            gms = ds.groupby('mnth_of_yr_num').mean().gms.values
            seas_score = MinMaxScaler().fit_transform(gms.reshape(-1,1)).reshape(-1)
            out = pd.DataFrame(np.array([int(ds.catg_id.unique().item())] + seas_score.tolist(), dtype='object').reshape(1,-1), columns=['catg_id']+[i for i in range(1,self.args.FORECAST_SIZE+1)])
            return out
        self.scores = self.dp.groupby('catg_id').apply(calculate_seasonality).reset_index(drop=True)
        self.scores.catg_id = self.scores.catg_id.astype('int')
        self.scores.loc[:, self.scores.keys()[1:]] = self.scores.loc[:, self.scores.keys()[1:]].values.astype('float')

    def filter(self, gms_csv):
        # filtering with seasonality_score

        self.calculate() # calculating seasonality score
        TIME_RANGE = np.arange(self.args.FORECAST_SIZE) + 1
        self.filter_dict = defaultdict(list) #key: time, value: catg_idx's to remove

        # changing to T/F value
        self.scores.loc[:, TIME_RANGE.tolist()] = (self.scores.loc[:, TIME_RANGE.tolist()].values < self.args.THRESHOLD_SEASONALITY)

        # adding items to DROP: items with seasonality score under the threshold
        for idx in self.scores.catg_id.unique():
            ds = self.scores[self.scores.catg_id == idx]
            for timestep in TIME_RANGE[ds.loc[:, TIME_RANGE.tolist()].values.reshape(-1).astype('bool')]:
                self.filter_dict[timestep].append(int(round(ds.catg_id.values[0])))

        for counter, (timestep, gms_tab) in enumerate(zip(self.args.TIME_ORDER, gms_csv)):
            drop_ids = []
            for id in self.filter_dict[timestep]:
                try:
                    drop_ids.append(gms_tab[gms_tab.catg_id==id].index.item())
                except ValueError:
                    pass
            gms_csv[counter] = gms_tab.drop(drop_ids).reset_index(drop=True)

        return gms_csv

# OPTIMIZER
class LpOptimizer:

    def __init__(self, args):
        self.args = args

    def solve(self, gms_tab, B, time, verbose=False):
        D = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25]) # discount rates
        N = gms_tab.catg_id.unique().shape[0]
        G = gms_tab[gms_tab.keys()[-len(D):]].values
        catgidxs = gms_tab.catg_id.values.astype('int')
        vecG = G.reshape(-1).astype('float')
        dG = np.repeat(D.reshape(1,-1), N, axis=0).reshape(-1) * vecG

        # declaring problem and variable
        program = LpProblem(name="GMS_Table_Optmization_{}".format(time), sense=LpMaximize)
        X = LpVariable.dicts("X", range(N*len(D)), cat='Binary')

        # objective function
        program += lpSum([X[i]*vecG[i] for i in range(N*len(D))])

        # equality constraint: choosing one entry per row of gms_table
        for i in range(N*len(D)):
            if i%len(D)==0:
                # equality constraint
                program += lpSum([X[i+j] for j in range(len(D))]) == 1

        # inequality constraint: budget constraint
        program += lpSum([dG[i]*X[i] for i in range(N*len(D))]) <= B

        # solve linear program
        program.solve()

        dsc_indicator = np.array([ [X[len(D)*i + j].varValue for j in range(len(D))] for i in range(N) ])
        dsc_index = np.where(dsc_indicator == 1)[1]
        dsc_choice = list(map(lambda x: D[x], dsc_index))

        # solution status
        obj = pulp.value(program.objective)
        spd = (dsc_indicator.reshape(-1) * dG).sum()

        if verbose:
            print("Status:", LpStatus[program.status])
            print("Objective: $ {:<.2f}".format(obj))
            print("Budget used: $ {:<.2f} out of $ {:<.2f}".format(spd, B))

        return obj, spd, dsc_choice, catgidxs

    def optmize(self, dp, gms_csv):
        TIME_BUDGET = np.array([self.args.BUDGET/self.args.FORECAST_SIZE]*self.args.FORECAST_SIZE) # array of budgets for each time

        obj_list = []
        spd_list = []
        dsc_list = []
        catgids_list = []

        for gms_tab, B, time in zip(gms_csv, TIME_BUDGET, self.args.TIME_ORDER):
            # for each timestep optimize and make a table of [items, time] promo calendar

            if gms_tab.shape[0] == 0:
                obj, spd, dsc_choice, catgids = 0, 0, [], []
            else:
                obj, spd, dsc_choice, catgids = self.solve(gms_tab, B, time)
            obj_list.append(obj)
            spd_list.append(spd)
            dsc_list.append(dsc_choice)
            catgids_list.append(catgids)

        # make the matrix first then make a dataframe with properly naming column keys
        promo_cal = np.zeros((dp.catg_id.unique().shape[0], self.args.FORECAST_SIZE))

        for counter in range(self.args.FORECAST_SIZE):
            for id, dsc in zip(catgids_list[counter], dsc_list[counter]):
                promo_cal[self.args.ID_TO_IDX[id], counter] = dsc

        output = np.empty((promo_cal.shape[0], self.args.FORECAST_SIZE+2), dtype='object')
        output[:,0] = [self.args.IDX_TO_ID[x] for x in range(promo_cal.shape[0])] # catg_id
        output[:,1] = [self.args.IDX_TO_NAME[x] for x in range(promo_cal.shape[0])] # catg_name
        output[:,2:] = promo_cal

        new_year_ind = np.where(np.array(self.args.TIME_ORDER)==1)[0].item() # index where the new year starts
        year_order = np.array([self.args.curr_year]*self.args.FORECAST_SIZE) # year order
        for i in range(new_year_ind, self.args.FORECAST_SIZE): # update the year order vector
            year_order[i] += 1
        time_keys = ['year_'+str(y)+'_month_'+str(m) for y, m in zip(year_order, self.args.TIME_ORDER)] # pandas column keys for times
        promo_cal = pd.DataFrame(output, columns = ['catg_id', 'catg_name'] + time_keys)

        return promo_cal

# GMS PREDICTOR
class Predictor:
    def __init__(self, df, args):
        self.df = df.copy()
        self.args = args

        self.criterion = nn.MSELoss()
        self.test_crit = nn.L1Loss() # MAE

        self.train_losses = []
        self.test_performance = []
        self.test_losses = []

        self.wmape_3hist = [] # logging high, mid, low at each step
        self.smape_3hist = []
        self.cpears_3hist = []
        self.cspear_3hist = []

        self.calibration_dict_time = {}
        self.calibration_dict_item = {}
        self.calibration_all = None

        def smape_func(pred, targ):
            if type(pred) == np.ndarray:
                val =  abs(pred-targ)/(abs(pred) + abs(targ))
                nan_mask = np.isnan(val)
            else:
                val = (pred-targ).abs()/(pred.abs() + targ.abs())
                nan_mask = torch.isnan(val)
            return val[~nan_mask].mean()

        def wmape_func(pred, targ, args, modify = True):
            if modify:
                if type(targ) == np.ndarray:
                    tt = np.copy(targ) + args.GMS_MIN/2
                else:
                    tt = targ.clone() + args.GMS_MIN/2
            else:
                tt = targ

            if type(targ) == np.ndarray:
                val = abs((pred - targ)/tt)
            else:
                val = ((pred - targ)/tt).abs()
            weight = targ/targ.sum()
            return (val*weight).sum()

        self.smape_func = smape_func
        self.wmape_func = wmape_func

    def aggregate(self):

        print('Aggregating data...')

        ID_TO_IDX = {}
        IDX_TO_ID = {}
        IDX_TO_NAME = {}
        NAME_TO_IDX = {}
        ID_TO_NAME = {}
        NAME_TO_ID = {}

        if self.args.TAXONOMY_LEVEL == 'catg':
            self.args.TAX_ID = 'catg_id'
            self.args.TAX_NAME = 'catg_name'
        elif self.args.TAXONOMY_LEVEL == 'subcatg':
            self.args.TAX_ID = 'subcatg_id'
            self.args.TAX_NAME = 'subcatg_name'

        self.df_name_date = self.df[pd.Index([self.args.TAX_ID]+self.df.keys()[-3:].tolist())]
        self.df_name_date = self.df_name_date.groupby(self.args.TAX_ID).apply(lambda x: x.iloc[0]).reset_index(drop=True)

        del self.df[self.args.TAX_NAME]
        del self.df['start_dt']
        del self.df['end_dt']

        if self.args.DATA_AGGREGATED:
            if self.args.TAXONOMY_LEVEL == 'catg':
                PATH_DATA_AGG = 'https://raw.githubusercontent.com/dykim1222/gmsdata/master/catg_mnth_agg.csv'
            elif self.args.TAXONOMY_LEVEL == 'subcatg':
                PATH_DATA_AGG = '/Users/dkim/Desktop/cleaning/data/subcatg_wk_agg.csv'
            self.dp = pd.read_csv(PATH_DATA_AGG)
            if self.args.DEBUG:
                self.dp = self.dp.iloc[:3000]

        else: # aggregate the data
            if self.args.TAXONOMY_LEVEL == 'catg':
                int_sep = 5
            elif self.args.TAXONOMY_LEVEL == 'subcatg':
                int_sep = 6
            def aggregate_func(x):
                # for a fixed item and time, if there are multiple transaction data for different discount rates,
                #     aggregate these data by summing GMS values and taking weighted average of discount rates.
                if x.shape[0] == 1:
                    out = pd.DataFrame([(x.iloc[0].values.astype('int')[:int_sep]).tolist() + \
                                        [x.site_promo_dsc_amt.values[0]] + \
                                        [x.holiday_ind.unique()[0]] + \
                                        [x.gms.values[0]]], \
                                       columns=x.keys())
                    return out
                gms_sum = x.gms.sum()
                if gms_sum == 0: # prevent NaNs
                    out = pd.DataFrame([(x.iloc[0].values.astype('int')[:int_sep]).tolist() + \
                                        [x.site_promo_dsc_amt.values[0]] + \
                                        [x.holiday_ind.unique()[0]] + \
                                        [x.gms.sum()]], \
                                       columns=x.keys())
                else:
                    weights = (x.gms/x.gms.sum()).values
                    out = pd.DataFrame([(x.iloc[0].values.astype('int')[:int_sep]).tolist() + \
                                        [x.site_promo_dsc_amt.values @ weights] + \
                                        [x.holiday_ind.unique()[0]] + \
                                        [x.gms.sum()]], \
                                       columns=x.keys())
                return out

            if self.args.TAXONOMY_LEVEL == 'catg':
                vec_sep = [self.args.TAX_ID, 'cal_yr_num', 'mnth_of_yr_num']
                name_agg = 'catg_mnth_agg.csv'
            elif self.args.TAXONOMY_LEVEL == 'subcatg':
                vec_sep = [self.args.TAX_ID, 'cal_yr_num', 'promo_wk_num']
                name_agg = 'subcatg_wk_agg.csv'

            self.dp = self.df.groupby(vec_sep).apply(aggregate_func).reset_index(drop=True)
            self.dp = self.dp.sort_values(vec_sep)
            self.dp.to_csv(self.args.PATH_SAVE + name_agg, index=False)

        pdb.set_trace()

        # merging start and end dates as features
        self.dp = self.dp.merge(self.df_name_date, on='catg_id', how='left')

        # removing NaN rows
        null_rows_index = self.dp[(pd.isnull(self.dp).catg_name)].index
        self.dp = self.dp.drop(null_rows_index)

        # parsing month and year from start/end dates
        start_col = np.array([ [int(date_str[:4]), int(date_str[5:7])] for date_str in self.dp.start_dt.values ])
        end_col = np.array([ [int(date_str[:4]), int(date_str[5:7])] for date_str in self.dp.end_dt.values ])

        self.dp['start_year'] = start_col[:,0]
        self.dp['start_mnth'] = start_col[:,1]
        self.dp['end_year'] = end_col[:,0]
        self.dp['end_mnth'] = end_col[:,1]

        del self.dp['start_dt']
        del self.dp['end_dt']
        del self.dp['catg_name']

        self.args.GMS_MIN = self.dp[self.dp.gms>0].gms.min()
        self.args.GMS_MAX = self.dp.gms.max()


        # DATA CLEANING
        # 1. REMOVING first month and last month
        self.dp.loc[:, self.dp.keys()[:5]] = self.dp.loc[:, self.dp.keys()[:5]].round().astype('int32')
        starttime = self.dp[self.dp.cal_yr_num == self.dp.cal_yr_num.min()].mnth_of_yr_num.min()
        endtime = self.dp[self.dp.cal_yr_num == self.dp.cal_yr_num.max()].mnth_of_yr_num.max()
        curr_year = self.dp.cal_yr_num.max()
        self.dp = self.dp.drop(self.dp[(self.dp.cal_yr_num == self.dp.cal_yr_num.min()) & (self.dp.mnth_of_yr_num == starttime)].index).reset_index(drop=True)
        self.dp = self.dp.drop(self.dp[(self.dp.cal_yr_num == self.dp.cal_yr_num.max()) & (self.dp.mnth_of_yr_num == endtime)].index).reset_index(drop=True)
        self.args.endtime = endtime
        self.args.curr_year = curr_year

        # 2. REMOVING OUTLIER
        how_many_to_remove = self.dp[self.dp.gms>self.args.THRESHOLD_GMS_OUTLIER].shape[0]
        remove_ids_list = []
        args_gms = np.argsort(-self.dp.gms.values)[:how_many_to_remove]
        remove_idx_list = []
        for og in args_gms:
            og_id = int(self.dp.iloc[og].catg_id)
            if og_id not in remove_ids_list:
                og_rows = self.dp[self.dp.catg_id == og_id]
                remove_idx_list += og_rows.index.values.tolist()
                remove_ids_list += [og_id]
        self.dp = self.dp.drop(remove_idx_list).reset_index(drop=True)
        self.dp_seas = self.dp.copy()

        # maps between name, id, idx's
        for idx, id in enumerate(self.dp.catg_id.unique()):

            item = self.df_name_date[self.df_name_date.catg_id == id]

            ID_TO_IDX[item.catg_id.item()] = idx
            IDX_TO_ID[idx] = item.catg_id.item()

            IDX_TO_NAME[idx] = item.catg_name.item()
            NAME_TO_IDX[item.catg_name.item()] = idx

            ID_TO_NAME[item.catg_id.item()] = item.catg_name.item()
            NAME_TO_ID[item.catg_name.item()] = item.catg_id.item()

        # TOKENIZING
        def catg_id_to_idx(x):
            return ID_TO_IDX[x]
        self.dp.catg_id = self.dp.catg_id.apply(catg_id_to_idx)
        # print('Tokeninzing TAXONOMY ID -> TAXONOMY IDX')

        # self.dp has catg_id as indexes
        # self.dp_sesas has catg_id as ids

        self.args.ID_TO_IDX = ID_TO_IDX
        self.args.IDX_TO_ID = IDX_TO_ID
        self.args.IDX_TO_NAME = IDX_TO_NAME
        self.args.NAME_TO_IDX = NAME_TO_IDX
        self.args.ID_TO_NAME = ID_TO_NAME
        self.args.NAME_TO_ID = NAME_TO_ID

        self.args.VOCAB_SIZE = len(ID_TO_IDX.keys())
        self.args.EMBEDDING_DIM = int(round(self.args.VOCAB_SIZE**(0.3)))


        print('Aggregating data done...')

    def preprocess(self):

        print('Preprocessing data...')

        cyclical_keys_periods = [4, 12, 3, 12, 12]
        cyclical_keys = self.dp.keys()[[2, 3, 4, 9, 11]]
        numeric_keys = self.dp.keys()[[1, 8, 10]]

        self.scaler_label_dict = {}

        # scaling year features
        scaler = MinMaxScaler()
        scaler.fit(self.dp['cal_yr_num'].values.reshape(-1,1))
        self.dp.loc[np.arange(self.dp.shape[0]), numeric_keys] = scaler.transform(self.dp[numeric_keys].values)

        # INDIVIDUAL SCALING of GMS
        for idx in range(len(self.dp.catg_id.unique())):
            ds = self.dp[self.dp.catg_id == idx]
            scaler_label = MinMaxScaler()
            self.dp.loc[ds.index, 'gms'] = scaler_label.fit_transform(ds.gms.values.reshape(-1,1)).reshape(-1)
            self.scaler_label_dict[idx] = scaler_label

        # ENCODING CYCLICAL VARIABLES
        # transform cyclic variables -> points on circles
        def trig_featurizer(x, L):
            #INPUT: integer in [1, L]
            #OUTPUT: cos, sin value on the circle
            return np.array([np.cos(2*math.pi/L*x), np.sin(2*math.pi/L*x)]).T
        # SIN/COS featurizer
        for period, key in zip(cyclical_keys_periods, cyclical_keys):
            dpk = self.dp[key].values
            dpk = trig_featurizer(dpk, period)
            self.dp[key+'_cos'] = dpk[:,0]
            self.dp[key+'_sin'] = dpk[:,1]
            del self.dp[key]
        # # ONEHOTENCODER
        # for key in cyclical_keys:
        #     enc = OneHotEncoder()
        #     enc.fit(dp[key].values.reshape(-1,1))
        #     res = enc.transform(dp[key].values.reshape(-1, 1)).toarray()
        #     for i, kkey in enumerate(enc.get_feature_names([key])):
        #         dp[kkey] = res[:,i]
        #     del dp[key]

        # move gms column to the end
        temp = self.dp.gms.values
        del self.dp['gms']
        self.dp['gms'] = temp

        print('Preprocessing data done...')

    def generate_dataset(self):
        # Make the table into sequential data

        print('Generating dataset...')

        if self.args.DATA_CREATED:
            # LOAD THE DATA
            train_data_load = torch.load(self.args.PATH_SAVE + 'train_data_save.pt')
            test_data_load = torch.load(self.args.PATH_SAVE + 'test_data_save.pt')
            self.trainX, self.trainY, self.train_dsc = train_data_load['x'], train_data_load['y'], train_data_load['d']
            self.testX, self.testY, self.test_dsc = test_data_load['x'], test_data_load['y'], test_data_load['d']

        else:
            # CREATE THE DATA
            self.trainX = []
            self.train_dsc = []
            self.trainY = []
            self.testX = []
            self.test_dsc = []
            self.testY = []

            for idx in (self.args.IDX_TO_ID.keys()):
                dd = self.dp[self.dp.catg_id == idx].reset_index(drop=True)
                length = dd.shape[0]
                test_set_size = int(round((length-self.args.FORECAST_SIZE+1 - self.args.CONTEXT_SIZE)*self.args.TEST_SET_SIZE))

                for i in range(self.args.CONTEXT_SIZE, length-self.args.FORECAST_SIZE+1):
                    # INPUT: x_t = [Time features at time t, Gms_t, discount_{t+1}]
                    # LABEL: Gms_{t+1}
                    sam = torch.from_numpy(dd.iloc[i-self.args.CONTEXT_SIZE:i, :].values).float()
                    future_dsc = torch.from_numpy(dd.iloc[i:i+self.args.FORECAST_SIZE,:].site_promo_dsc_amt.values).float()
                    future_gms = torch.from_numpy(dd.iloc[i:i+self.args.FORECAST_SIZE,:].gms.values).float()

                    if i < length - self.args.FORECAST_SIZE+1 - test_set_size:
                        # append to train set
                        self.trainX.append(sam.unsqueeze(0))
                        self.train_dsc.append(future_dsc.unsqueeze(0))
                        self.trainY.append(future_gms.unsqueeze(0))
                    else:
                        # append to test set
                        self.testX.append(sam.unsqueeze(0))
                        self.test_dsc.append(future_dsc.unsqueeze(0))
                        self.testY.append(future_gms.unsqueeze(0))

            # SAVE THE DATA
            torch.save({'x': self.trainX, 'y': self.trainY, 'd': self.train_dsc}, self.args.PATH_SAVE+'train_data_save.pt')
            torch.save({'x':self.testX, 'y': self.testY, 'd':self.test_dsc}, self.args.PATH_SAVE+'test_data_save.pt')

        self.train_data = TensorDataset(torch.cat(self.trainX, dim = 0), torch.cat(self.trainY, dim=0), torch.cat(self.train_dsc, dim=0))
        self.train_loader = DataLoader(self.train_data, shuffle=True, batch_size=self.args.BATCH_SIZE, drop_last = True)
        # print('Train/Validation Split done!')

        print('Generating dataset done...')

    def train(self):

        print('Training the model...')

        # Setting common hyperparameters
        input_dim = self.trainX[0].shape[2] - 1 + self.args.EMBEDDING_DIM
        output_dim = 1
        # self.val_best = float('inf')
        self.val_best = 0

        if self.args.DEBUG:
            self.args.HIDDEN_DIM = 2
            self.args.LATENT_DIM = 2
            self.args.N_LAYERS = 2
            self.args.EPOCHS = 3
            self.args.NUM_KERNELS1 = 3
            self.args.NUM_KERNELS2 = 3

        # Instantiating the models
        if self.args.model_type == "GRU":
            model = GRUNet(input_dim, output_dim, self.args)
        elif self.args.model_type == "LSTM":
            model = LSTMNet(input_dim, output_dim, self.args)
        elif self.args.model_type == "LSTMCONV":
            model = LSTMConvNet(input_dim, output_dim, self.args)
        model.to(device)

        self.evaluate(model)

        # Defining loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True,  cooldown=10)
        print("Starting Training of {} model".format(self.args.model_type))
        self.epoch_times = []

        # Start training loop
        for epoch in range(1,self.args.EPOCHS+1):
            model.train()
            start_time = time.time()
            avg_loss = 0.
            counter = 0
            for x, label, dsc in (self.train_loader):
                h = model.init_hidden(self.args.BATCH_SIZE)
                counter += 1
                h = h.data if self.args.model_type == "GRU" else tuple([e.data for e in h])

                model.zero_grad()
                out = model(x.to(device), dsc.to(device), h)
                loss = self.criterion(out, label.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.args.CLIP_GRAD)
                optimizer.step()
                avg_loss += loss.item()
            current_time = time.time()
            print("Epoch {}/{} Done, \nTotal Loss: {:.<5f}".format(epoch, self.args.EPOCHS, avg_loss/len(self.train_loader)))

            self.evaluate(model)
            # scheduler.step(test_perf)
            self.train_losses.append(avg_loss/len(self.train_loader))
            self.epoch_times.append(current_time-start_time)

            if self.test_performance[-1] > self.val_best:
                self.val_best = self.test_performance[-1]
                save_data = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scale': self.scaler_label_dict,
                    'calib_time': self.calibration_dict_time,
                    'calib_item': self.calibration_dict_item,
                    'calib_all': self.calibration_all
                    }
                torch.save(save_data, self.args.PATH_MODEL_SAVE)
                self.saved_model = save_data
                print('Model Saved.')
            print('Best Validation Performance = {:<.4f}'.format(self.val_best))
            print("Time Elapsed for Epoch: {} seconds".format(str(current_time-start_time)))
            print('-'*60)

        # return model
        print("Total Training Time: {} seconds".format(str(sum(self.epoch_times))))

        print('Training the model done...')

    def evaluate(self, model, analysis=False):
        model.eval()
        outputs = []
        targets = []
        start_time = time.time()
        x, y, dsc = torch.cat(self.testX, dim=0), torch.cat(self.testY,dim=0), torch.cat(self.test_dsc,dim=0)
        h = model.init_hidden(x.shape[0])

        out = model(x.to(device), dsc.to(device), h)
        out = out.detach().cpu().numpy()

        # SCALING BACK TO ORIGINAL SCALE
        test_set_idx = x[:, 0, 0].long().numpy() # item indexes
        out = np.concatenate([self.scaler_label_dict[idx].inverse_transform(o.reshape(-1,1)).T for idx, o in zip(test_set_idx, out)])
        y = np.concatenate([self.scaler_label_dict[idx].inverse_transform(o.reshape(-1,1)).T for idx, o in zip(test_set_idx, y)])

        test_loss = self.test_crit(torch.from_numpy(out).to(device), torch.from_numpy(y).to(device))

        smape_list = []
        wmape_list = []
        cp_list = []
        cs_list = []

        # calibration all
        self.calibration_all = np.polyfit(out.reshape(-1), y.reshape(-1), 1)

        # calibration across time dimension
        for i in range(self.args.FORECAST_SIZE):

            line_slope, line_bias = np.polyfit(out[:,i], y[:,i], 1)
            self.calibration_dict_time[i] = (line_slope, line_bias)

            smape_list.append(self.smape_func(out[:,i], y[:,i]))
            wmape_list.append(self.wmape_func(out[:,i], y[:,i], self.args))
            cp_list.append(np.corrcoef(out[:,i], y[:,i])[0,1])
            cs_list.append(spearmanr(out[:,i], y[:,i])[0])

        # calibration across item dimension
        out_calib = out.reshape(len(set(test_set_idx)), -1)
        y_calib = y.reshape(len(set(test_set_idx)), -1)
        for counter, idx in enumerate(test_set_idx.reshape(len(set(test_set_idx)),-1)[:,0]):
            line_slope, line_bias = np.polyfit(out_calib[counter], y_calib[counter], 1)
            self.calibration_dict_item[idx] = (line_slope, line_bias)


        if not analysis:
            self.wmape_3hist.append([100*np.min(wmape_list), 100*np.median(wmape_list), 100*np.max(wmape_list)])
            self.smape_3hist.append([100*np.min(smape_list), 100*np.median(smape_list), 100*np.max(smape_list)])
            self.cpears_3hist.append([np.min(cp_list), np.median(cp_list), np.max(cp_list)])
            self.cspear_3hist.append([np.min(cs_list), np.median(cs_list), np.max(cs_list)])

            print('wMAPE: Min = {:<.1f}%, Median = {:<.1f}%, Max = {:<.1f}%'.format(*self.wmape_3hist[-1]))
            print('sMAPE: Min = {:<.1f}%, Median = {:<.1f}%, Max = {:<.1f}%'.format(*self.smape_3hist[-1]))
            print('C_Pears: Min = {:<.3f}, Median = {:<.3f}, Max = {:<.3f}'.format(*self.cpears_3hist[-1]))
            print('C_Spear: Min = {:<.3f}, Median = {:<.3f}, Max = {:<.3f}'.format(*self.cspear_3hist[-1]))

        wMAPE = np.median(wmape_list)
        corr_pears = np.median(cp_list)
        corr_spear = np.median(cs_list)

        test_perf = corr_spear
        self.test_performance.append(test_perf)
        self.test_losses.append(test_loss.item())

    def infer(self):
        print('Inference...')

        # Loading best model for Evaluation
        input_dim = self.trainX[0].shape[2] - 1 + self.args.EMBEDDING_DIM
        output_dim = 1
        if self.args.model_type == "GRU":
            model = GRUNet(input_dim, output_dim, self.args)
        elif self.args.model_type == "LSTM":
            model = LSTMNet(input_dim, output_dim, self.args)
        elif self.args.model_type == "LSTMCONV":
            model = LSTMConvNet(input_dim, output_dim, self.args)
        model.to(device)

        loaded = torch.load(self.args.PATH_MODEL_SAVE)
        model.load_state_dict(loaded['model'])
        self.scaler_label_dict = loaded['scale']
        self.calibration_all = loaded['calib_all']
        self.calibration_dict_time = loaded['calib_time']
        self.calibration_dict_item = loaded['calib_item']


        # INFERENCE
        inputX = []
        for idx in (self.args.IDX_TO_ID.keys()):
            dd = self.dp[self.dp.catg_id == idx].reset_index(drop=True)
            sam = torch.from_numpy(dd.iloc[-self.args.CONTEXT_SIZE:, :].values).float()
            inputX.append(sam.unsqueeze(0))

        # For each discount rate
        dsc_rates = [0., 0.05, 0.1, 0.15, 0.2, 0.25]
        input_dsc = []

        for dsc in dsc_rates:
            future_dsc = torch.Tensor([dsc]*self.args.FORECAST_SIZE).float()
            input_dsc.append(future_dsc.unsqueeze(0))

        if self.args.DEBUG:
            inputX = torch.cat(inputX[:-1])
        else:
            inputX = torch.cat(inputX)
        input_dsc = torch.cat(input_dsc)

        # Initialize gms_tensor and predict
        # After prediction: scaling back to original and calibrate
        gms_tensor = []
        model.eval()
        pdb.set_trace()
        for dsc in input_dsc:
            h = model.init_hidden(inputX.shape[0])
            out = model(inputX.to(device), dsc.unsqueeze(0).expand(inputX.shape[0],-1).to(device), h)
            out = out.detach().cpu().numpy()

            # original scale
            pred_set_idx = inputX[:, 0, 0].long().numpy()
            out = np.concatenate([self.scaler_label_dict[idx].inverse_transform(o.reshape(-1,1)).T for idx, o in zip(pred_set_idx, out)])

            # 4 CALIBRATION VARIATIONS: item, time, all, none
            if self.args.CALIBRATION_MODE == 'all': # calib all
                print('Calibration all')
                out = (self.calibration_all[1] + self.calibration_all[0]*out).reshape(self.args.VOCAB_SIZE, -1)
            elif self.args.CALIBRATION_MODE == 'time': # calibrate across time
                print('Calibration time')
                out = np.concatenate([((out[:,i]*self.calibration_dict_time[i][0]) + self.calibration_dict_time[i][1]).reshape(self.args.VOCAB_SIZE, 1)  for i in range(self.args.FORECAST_SIZE)], axis = 1)
            elif self.args.CALIBRATION_MODE == 'item': # calibrate across items
                print('Calibration item')
                out = np.concatenate([(o*self.calibration_dict_item[idx][0] + self.calibration_dict_item[idx][1]).reshape(1,-1) for idx, o in zip(pred_set_idx, out)])
            elif self.args.CALIBRATION_MODE == 'none':
                print('Calibration none')
                pass

            gms_tensor.append(np.expand_dims(out, 1))
        pdb.set_trace()

        gms_tensor = np.concatenate(gms_tensor, axis=1)
        print('Negative entries after calibration {:<.2f} %'.format(100*(gms_tensor<0).mean()))

        # plt.hist(gms_tensor[gms_tensor<0].reshape(-1), bins=20)
        # plt.title('GMS Distribution of Negative entries')
        # plt.xlabel('GMS')
        # plt.show()

        gms_tensor = np.maximum(gms_tensor, 0) # gms_tensor.shape # [num_items, num_dsc_rates, num_timesteps]

        # TRANSFORMING INTO PANDAS DATAFRAMES
        catgid = np.array(list(map(lambda x: self.args.IDX_TO_ID[x], pred_set_idx))).astype(int)
        catgid = np.expand_dims(catgid, 1)

        catgname = np.array([self.args.ID_TO_NAME[cid] for cid in catgid.reshape(-1).tolist()])
        catgname = np.expand_dims(catgname, 1)

        TIME_ORDER = []
        gms_csv = []
        END_TIME_SCALE = 12 if self.args.TIME_SCALE == 'month' else 48
        curryear = self.args.curr_year

        # generating gms tensor
        for i in range(self.args.FORECAST_SIZE):

            gms_tab = gms_tensor[:,:,i]

            csv = pd.DataFrame(np.concatenate((catgid, catgname, gms_tab), axis = 1),
                        columns = ['catg_id', 'catg_name']+['gms_pred_{}'.format(x) for x in [0,5,10,15,20,25]])
            csv.catg_id = csv.catg_id.astype('int')
            csv.loc[:, csv.keys()[-6:]] = csv.loc[:, csv.keys()[-6:]].values.astype('float')

            time_num = (self.args.endtime+i)%END_TIME_SCALE
            if time_num == 0:
                time_num = END_TIME_SCALE

            # csv.to_csv('year_{}_time_{}.csv'.format(curryear, time_num), index = False)
            gms_csv.append(csv)

            inc = ((gms_tab[:,-1] - gms_tab[:,0]) > 0).mean()
            zer = ((gms_tab[:,-1] - gms_tab[:,0]) == 0).mean()
            print('Year {} Month {} Increasing: {:<.2f}, Zero: {:<.2f}, Decreasing: {:<.2f}'.format(curryear, time_num, inc, zer, 1-inc-zer))

            if time_num == END_TIME_SCALE:
                curryear += 1
            TIME_ORDER.append(time_num)

        self.args.TIME_ORDER = TIME_ORDER

        print('Inference done...')

        self.gms_csv = gms_csv
        return gms_csv

    def apply_season_filter(self, gms_csv):

        print('Filtering predictions...')

        self.seasoner = Seasonalizer(self.dp_seas, self.args)

        print('Filtering predictions done...')

        return self.seasoner.filter(gms_csv)

    def postprocess(self, gms_csv):

        print('Postprocessing...')

        # drop constant items  OR zero out once starting to decrease.

        for counter, gms_tab in enumerate(gms_csv):
            slopes = gms_tab[gms_tab.keys()[-6:]].values[:, 1:].astype('float') - gms_tab[gms_tab.keys()[-6:]].values[:, :-1].astype('float')
            slopes = (slopes > 0)
            drop_idxs = []
            for i in range(gms_tab.shape[0]):
                if not slopes[i].any(): # all Falses: all non-increasing
                    drop_idxs.append(i)
                else:
                    for kounter, tval in enumerate(slopes[i]): # zero-ing out non-increasing parts
                        if not tval:
                            gms_tab.loc[i, gms_tab.keys()[-(6-(kounter+1)):]] = 0
                            break
            gms_csv[counter] = gms_tab.drop(drop_idxs).reset_index(drop=True)

        print('Postprocessing done...')

        return gms_csv

    def optimize(self, gms_csv):

        print('Optimizing...')

        optimizer = LpOptimizer(self.args)
        promo_cal = optimizer.optmize(self.dp, gms_csv)
        promo_cal.to_csv(self.args.PATH_SAVE+'promo_cal.csv', index=False)

        print('Optimizing done...')

        return promo_cal
