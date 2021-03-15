import numpy as np
from collections import defaultdict


# Agrega a sequencia de rewards para cada estimulo em uma lista
def aggregate_stimuli(session):

    st_rt_train = defaultdict(list)
    st_rt_test = defaultdict(list)

    for trial in session.train_set:
        st, ac, rt, bs = trial
        st_rt_train[st].append(rt)

    for trial in session.test_set:
        st, ac, rt, bs = trial
        st_rt_test[st].append(rt)

    st_rt_train = {st: np.array(rt_series) for st, rt_series in st_rt_train.items()}
    st_rt_test = {st: np.array(rt_series) for st, rt_series in st_rt_test.items()}

    return st_rt_train, st_rt_test
    
    
# Calcula a sequencia com o reward médio para cada estímulo 
def average_stimuli_probs(session_list):

    st_rt_avg_train = defaultdict(list)
    st_rt_avg_test = defaultdict(list)

    for session in session_list:

        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, rt_series in st_rt_train.items():
            st_rt_avg_train[st].append(rt_series)

        for st, rt_series in st_rt_test.items():
            st_rt_avg_test[st].append(rt_series)

    st_rt_avg_train = {st: np.mean(rts_list, axis=0) for st, rts_list in st_rt_avg_train.items()}
    st_rt_avg_test = {st: np.mean(rts_list, axis=0) for st, rts_list in st_rt_avg_test.items()}

    return st_rt_avg_train, st_rt_avg_test


# Calcula a sequencia com o reward médio para cada tamanho de bloco 
def average_block_probs(session_list):

    bs_rt_avg_train = defaultdict(list)
    bs_rt_avg_test = defaultdict(list)

    for session in session_list:

        bs_rt_train = defaultdict(list)
        bs_rt_test = defaultdict(list)

        #print("Aggragating session " + str(session.caseid))
        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, rt_series in st_rt_train.items():
            bs = session.response_map.loc[st]['Block']
            bs_rt_train[bs].append(rt_series)

        for st, rt_series in st_rt_test.items():
            bs = session.response_map.loc[st]['Block']
            bs_rt_test[bs].append(rt_series)
 
        for bs, rts_list in bs_rt_train.items():
            bs_rt_avg_train[bs].append(np.mean(rts_list, axis=0))

        for bs, rts_list in bs_rt_test.items():
            bs_rt_avg_test[bs].append(np.mean(rts_list, axis=0))

    bs_rt_avg_train = {bs: np.mean(rts_list, axis=0) for bs, rts_list in bs_rt_avg_train.items()}
    bs_rt_avg_test = {bs: np.mean(rts_list, axis=0) for bs, rts_list in bs_rt_avg_test.items()}

    return bs_rt_avg_train, bs_rt_avg_test
    

# Calcula a taxa de reward assintotica para cada tamanho de bloco 
def aggregate_as_probs(session_list, last_train):

    bs_as_pairs_train = []
    bs_as_pairs_test = []
    
    for session in session_list:

        bs_as_train = defaultdict(list)
        bs_as_test = defaultdict(list)

        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, series in st_rt_train.items():
            bs = session.response_map.loc[st]['Block']
            as_prob = np.mean(series[-last_train:])
            bs_as_train[bs].append(as_prob)
        
        for st, series in st_rt_test.items():
            bs = session.response_map.loc[st]['Block']
            as_prob = np.mean(series)
            bs_as_test[bs].append(as_prob)

        bs_as_pairs_train.append((np.mean(bs_as_train[3]), np.mean(bs_as_train[6])))
        bs_as_pairs_test.append((np.mean(bs_as_test[3]), np.mean(bs_as_test[6])))

    return bs_as_pairs_train, bs_as_pairs_test


# Count perseverance errors
def count_perseverance(session_list):

    pers_count = {}
    for session in session_list:
        pers_count[session.caseid] = 0
        past_errs = defaultdict(set)
        for trial in session.train_set:
            st, ac, rt, bs = trial
            if rt == 0:
                if ac in past_errs[st]:
                    pers_count[session.caseid] +=1
                else:
                    past_errs[st].add(ac)              
    return pers_count

    
