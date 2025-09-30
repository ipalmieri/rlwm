import os
from collections import Counter
import numpy as np
import pandas as pd


DATA_HEADER = ['Stimulus_Pair', 'response', 'correct', 'Block']

class DataSession():
    '''Base class containing experiment data'''
    def __init__(self, caseid):
        self.caseid = caseid
        self.possible_stimuli = []
        self.possible_actions = []
        self.response_map = {}
        self.train_set = []
        self.test_set = []
        self.st_maxlen_train = 0
        self.st_maxlen_test = 0

    def get_reward(self, stimulus, action):
        ac_correct = self.response_map.loc[stimulus]['correct_response']
        return 1. if action == ac_correct else 0.

    def get_blocksize(self, stimulus):
        return self.response_map.loc[stimulus]['Block']

    @classmethod
    def from_sequence(cls, caseid, train_set, test_set, response_dict):
        session = cls(caseid)
        session.possible_stimuli = list(set([trial[0] for trial in train_set]))
        session.possible_actions = list(set([trial[1] for trial in train_set]))
        session.train_set = train_set
        session.test_set = test_set
        # Create map stimulus to response and block size
        blocksize_dict = {trial[0]: trial[3] for trial in train_set}
        if not isinstance(response_dict, dict):
            response_dict = {trial[0]: trial[1] for trial in train_set if trial[2] > 0}
        block_col = [blocksize_dict[st] for st in session.possible_stimuli]
        respo_col = [response_dict[st] for st in session.possible_stimuli]
        session.response_map = pd.DataFrame(zip(session.possible_stimuli, respo_col, block_col), columns=['Stimulus_Pair', 'correct_response', 'Block'])
        session.response_map.set_index['Stimulus_Pair']
        return session

    @classmethod
    def from_df(cls, caseid, df_train, df_test):
        session = cls(caseid)
        session.possible_stimuli = df_train['Stimulus_Pair'].unique().tolist()
        session.possible_actions = df_train['correct_response'].unique().tolist()
        session.response_map = df_train.groupby(['Stimulus_Pair', 'correct_response', 'Block']).size().reset_index()
        session.response_map = session.response_map[['Stimulus_Pair', 'correct_response', 'Block']].set_index('Stimulus_Pair')
        session.train_set = df_train[DATA_HEADER].values.tolist()
        session.test_set = df_test[DATA_HEADER].values.tolist()
        session.st_maxlen_train = (
            np.max(list(Counter(s[0] for s in session.train_set).values()))
            if session.train_set else 0
        )
        session.st_maxlen_test = (
            np.max(list(Counter(s[0] for s in session.test_set).values()))
            if session.test_set else 0
        )
        return session


class tsDataSession(DataSession):
    '''Extended class adding response time to data series'''
    def __init__(self, caseid):
        super().__init__(caseid)
        self.train_ts = []
        self.test_ts = []

    @classmethod
    def from_df(cls, caseid, df_train, df_test):
        session = super(tsDataSession, cls).from_df(caseid, df_train, df_test)
        session.train_ts = df_train['response_time'].tolist()
        session.test_ts = df_test['response_time'].tolist()
        return session


