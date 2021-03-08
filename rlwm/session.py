import os
from collections import Counter
import numpy as np
import pandas as pd

TRAINFILE_SUFFIX = '-RLWM-Parte 1_c'
TESTFILE_SUFFIX = '-RLWM-Parte 2_c'


class DataSession():

  def __init__(self, caseid, df_train, df_test):
    self.caseid = caseid
    self.possible_stimuli = df_train['Stimulus_Pair'].unique().tolist()
    self.possible_actions = df_train['correct_response'].unique().tolist()
    self.response_map = df_train.groupby(['Stimulus_Pair', 'correct_response', 'Block']).size().reset_index()
    self.response_map = self.response_map[['Stimulus_Pair', 'correct_response', 'Block']].set_index('Stimulus_Pair')
    self.train_set = list(zip(df_train['Stimulus_Pair'], df_train['response'], df_train['correct'], df_train['Block']))
    self.test_set = list(zip(df_test['Stimulus_Pair'], df_test['response'], df_test['correct'], df_test['Block']))
    self.st_maxlen_train = np.max([c for s, c in Counter([s[0] for s in self.train_set]).items()])
    self.st_maxlen_test = np.max([c for s, c in Counter([s[0] for s in self.test_set]).items()])

  def get_reward(self, stimulus, action):
    ac_correct = self.response_map.loc[stimulus]['correct_response']
    return 1. if action == ac_correct else 0.




def load_dataset(caseid, data_path):
  train_filename = str(caseid) + TRAINFILE_SUFFIX + '.csv'
  test_filename = str(caseid) + TESTFILE_SUFFIX + '.csv'
  df_train = pd.read_csv(os.path.join(data_path, train_filename), sep=',')
  df_test = pd.read_csv(os.path.join(data_path, test_filename), sep=',')
  return df_train, df_test


  

def load_session(caseid):
  df_train, df_test = load_dataset(caseid)
  ds = DataSession(caseid, df_train, df_test)
  return ds
  
