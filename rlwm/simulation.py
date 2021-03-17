import numpy as np
from collections import defaultdict
import copy
from . import models


def simulate_model_session(model, session):

    out_train = []
    for trial in session.train_set:
        st, ac, rt, bs = trial
        ac_i = model.get_action(st, bs, test=False)
        rt_i = session.get_reward(st, ac_i)
        out_train.append((st, ac_i, rt_i, bs))
        model.learn_sample(st, ac, rt, bs)

    out_test = []
    for trial in session.test_set:
        st, ac, rt, bs = trial
        ac_i = model.get_action(st, bs, test=True)
        rt_i = session.get_reward(st, ac_i)
        out_test.append((st, ac_i, rt_i, bs))
    
    model_session = copy.deepcopy(session)
    model_session.train_set = out_train
    model_session.test_set = out_test
    return model_session


def simulate_model(model_func, params, session):
    
    model = models.get_model(model_func, params)
    model.init_model(session.possible_stimuli, session.possible_actions)
    model_session = simulate_model_session(model, session)
    return model_session
        

def average_rts_dict(rts_dict_list):

    rt_avg = defaultdict(list)
    for rt_dict in rts_dict_list:
        for k, rt_series in rt_dict.items():
            rt_avg[k].append(rt_series)
    rt_avg = {k: np.mean(rts_list, axis=0) for k, rts_list in rt_avg.items()}
    return rt_avg
    
    
def simulate_model_curves(model_func, params, session, epochs):

    model_session_list = []
    for i in range(epochs):
        model_session = simulate_model(model_func, params, session)
        model_session_list.append(model_session)

    st_rt_avg_train, st_rt_avg_test = average_stimuli_probs(model_session_list)
    bs_rt_avg_train, bs_rt_avg_test = average_block_probs(model_session_list)

    return st_rt_avg_train, st_rt_avg_test, bs_rt_avg_train, bs_rt_avg_test
    


def simulate_all_sessions(model_func, param_dict, session_list, epochs):

    st_rt_avg_train = []
    st_rt_avg_test = []
    bs_rt_avg_train = []
    bs_rt_avg_test = []

    for session in tqdm(session_list, position=0):

        params = param_dict[session.caseid]
        curves = simulate_model_curves(model_func, params, session, epochs)
        
        st_rt_avg_train.append(curves[0])
        st_rt_avg_test.append(curves[1])
        bs_rt_avg_train.append(curves[2])
        bs_rt_avg_test.append(curves[3])
        
    st_rt_avg_train = average_rts_dict(st_rt_avg_train)
    st_rt_avg_test = average_rts_dict(st_rt_avg_test)
    bs_rt_avg_train = average_rts_dict(bs_rt_avg_train)
    bs_rt_avg_test = average_rts_dict(bs_rt_avg_test)

    return st_rt_avg_train, st_rt_avg_test, bs_rt_avg_train, bs_rt_avg_test
    
    
