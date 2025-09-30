import os
import re
from collections import defaultdict
import pandas as pd
from rlwm.session import tsDataSession


FN_FORMATS = {
    "C":        "{caseid}-RLWM-Parte {part}_c",
    "HYRO":     "{caseid}-RLWM-Parte {part}_HYRO",
    "SUB":      "^subject-{caseid}-(.+)$"
}

FN_EXT = ".csv"

DF_HEADER = ['Stimulus_Pair', 
             'response', 
             'correct_response', 
             'response_time',
             'Block']


def recalculate_blocks(df_train, df_test):

    st_colname = 'Stimulus_Pair'
    bl_colname = 'Block'
    st_pairs = pd.concat([df_train[st_colname], df_test[st_colname]])

    bl_map = defaultdict(set)
    get_root = lambda s: re.sub(r"\d", "", s)
    get_bs = lambda s: len(bl_map[get_root(s)])

    for st in st_pairs.to_list():

        root_st = get_root(st)        
        bl_map[root_st].add(st)

    df_train[bl_colname] = df_train[st_colname].apply(get_bs)
    df_test[bl_colname] = df_test[st_colname].apply(get_bs)



def load_session_file(caseid, data_path, filename_train, filename_test=None):

    df_train = pd.DataFrame(columns=DF_HEADER)
    df_test = pd.DataFrame(columns=DF_HEADER)

    if filename_train:
        fullpath_train = os.path.join(data_path, filename_train)
        df_train_raw = pd.read_csv(fullpath_train, sep=',')
        df_train_cols = df_train.columns.intersection(df_train_raw.columns)
        df_train = df_train_raw[df_train_cols].copy()

    if filename_test:
        fullpath_test = os.path.join(data_path, filename_test)
        df_test_raw = pd.read_csv(fullpath_test, sep=',')
        df_test_cols = df_test.columns.intersection(df_test_raw.columns)
        df_test = df_test_raw[df_test_cols].copy()

    # Forced fixes for missing data
    if 'Block'not in df_train.columns.intersection(df_test.columns):
        recalculate_blocks(df_train, df_test)

    # Normalize responses to uppercase
    df_train['response'] = df_train['response'].str.upper()
    df_test['response'] = df_test['response'].str.upper()
    df_train['correct_response'] = df_train['correct_response'].str.upper()
    df_test['correct_response'] = df_test['correct_response'].str.upper()

    # Response 
    df_train["correct"] = (df_train["response"] == df_train["correct_response"])    
    df_test["correct"] = (df_test["response"] == df_test["correct_response"])    
    df_train['correct'] = df_train['correct'].astype(int)
    df_test['correct'] = df_test['correct'].astype(int)

    # Create session object
    ds = tsDataSession.from_df(caseid, df_train, df_test)

    return ds


def load_batch_folder(data_path):

    ret_batch = []
    fn_map = defaultdict(lambda: {'train': None, 'test': None}.copy())

    files_list = [fn for fn in os.listdir(data_path) if fn.endswith(FN_EXT)]

    for fn in files_list:
 
        cid = None

        for key, fn_mask in FN_FORMATS.items():
         
            # Check if it is a train file
            ptrn_train = re.compile(fn_mask.format(caseid=r'(\d+)', part="1"))

            fn_res_train = ptrn_train.match(fn)

            if fn_res_train:
                cid = fn_res_train.group(1)
                fn_map[cid]['train'] = fn
                continue

            # Check if it is a test file
            ptrn_test = re.compile(fn_mask.format(caseid=r'(\d+)', part="2"))

            fn_res_test = ptrn_test.match(fn)

            if fn_res_test:
                cid = fn_res_test.group(1)
                fn_map[cid]['test'] = fn
                continue
    
    for cid, fn_vals in fn_map.items():

            fn_train = fn_vals['train']
            fn_test = fn_vals['test']

            ret_batch.append(load_session_file(cid, data_path, fn_train, fn_test))

    return ret_batch



def load_batch(data_path, caseids=[]):

    if not len(caseids) > 0:
        return load_batch_folder(data_path)

    ret_batch = []
    files_list = [fn for fn in os.listdir(data_path) if fn.endswith(FN_EXT)]

    for cid in caseids:

        fn_train = None
        fn_test = None

        for key, fn_mask in FN_FORMATS.items():
            
            ptrn_train = re.compile(fn_mask.format(caseid=cid, part="1"))
            ptrn_test = re.compile(fn_mask.format(caseid=cid, part="2"))

            for fn in files_list:
                
                fn_res_train = ptrn_train.match(fn)
                fn_res_test = ptrn_test.match(fn)

                if fn_res_train:
                    fn_train = fn
                if fn_res_test:
                    fn_test = fn

            if fn_train and fn_test:
                break
        
        if fn_train or fn_test:

            if fn_test == fn_train:
                fn_test = None

            ret_batch.append(load_session_file(cid, data_path, fn_train, fn_test))
        
    return ret_batch


def load_session(caseid, data_path):
    return load_batch(data_path, [caseid])[0]
