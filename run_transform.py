import os
import pandas as pd
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.models as models
import rlwm.transformation as transformation
import params

RUN_BATCH='batch01'
RUN_SUFFIX='c'
RUN_CNR='beta_0-50'

BASE_PATH = '/srv/black/data/rlwm'
DATA_PATH = os.path.join(BASE_PATH, 'dados', RUN_BATCH)



CASEIDS = params.CASEIDS_BATCH01
#CASEIDS = CASEIDS[:5]



def main():

    # Load all datasets
    session_list = []
    for id in CASEIDS:
      ds = session.load_session(id, DATA_PATH, RUN_SUFFIX)
      session_list.append(ds)
    print(f'{len(session_list)} cases loaded')

    id_bs_rt_train, id_bs_rt_test = transformation.average_caseid_block_probs(session_list)

    df = pd.DataFrame()

    for caseid, bs_rt_list in id_bs_rt_train.items():
        df_caseid = pd.DataFrame.from_dict(bs_rt_list, orient='index')
        df_caseid.index.rename('block_size', inplace=True)
        df_caseid = df_caseid.reset_index()
        df_caseid['caseid'] = caseid
        df_caseid = df_caseid.melt(id_vars=['caseid', 'block_size'], var_name='iteration', value_name='P_correct')
        #df_caseid = df_caseid.set_index(['caseid', 'block_size', 'iteration'])
        df = df.append(df_caseid)

     df.to_excel('caseid_block_probls.xlsx')


if __name__ == '__main__':
    freeze_support()
    main()  
