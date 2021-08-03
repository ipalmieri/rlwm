import os
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.models as models
import rlwm.optsp as optsp
import rlwm.optho as optho
import rlwm.optimization as optimization
import rlwm.simulation as simulation
from collections import defaultdict
from tqdm import tqdm

BASE_PATH = '/srv/black/data/rlwm'
DATA_PATH = os.path.join(BASE_PATH, 'dados')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH = os.path.join(BASE_PATH, 'models/geral')
#MODEL_PATH = None
OPT_REPS = 30
OPT_EVALMAX = 1000

CASEIDS = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 17, 25, 26, 27, 29, 37, 49, 54, 57, 59, 62, 64, 66, 72, 76, 77, 79, 84, 91, 92, 94, 97, 102, 105, 110, 118, 127, 132, 153, 155, 159, 161, 164, 166, 172, 174, 175, 178, 179, 180, 181, 182, 184, 185, 187, 189, 195, 197, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 214, 215, 216, 217, 218, 220, 221, 223, 224, 226, 227]

#CASEIDS = CASEIDS[:5]
CASEIDS = [2]



def main():

    # Load all datasets
    session_list = []
    for id in CASEIDS:
      ds = session.load_session(id, DATA_PATH)
      session_list.append(ds)
    print(f'{len(session_list)} cases loaded')


    opt_modelfunc = models.model_rlwma
    opt_model_name = 'model_rlwmz'
    opt_solver = 'scipy'
    sim_epochs = 1
    
    p_sp = {}
    f_sp = {}

    print(f'Loading params for cases {[s.caseid for s in session_list]}')
    for s in session_list:
    
        p, f = optimization.get_model_params(opt_modelfunc, 
                                             s, 
                                             solver=opt_solver, 
                                             model_name=opt_model_name, 
                                             model_path=MODEL_PATH)
        p_sp[s.caseid] = p
        f_sp[s.caseid] = f



    param_dict = p_sp
    session_epoch = defaultdict(list)
    
    for s in tqdm(session_list):

        caseid = s.caseid
        for i in range(sim_epochs):
            session_epoch[caseid].append(simulation.simulate_session(opt_modelfunc, param_dict[caseid], s))






if __name__ == '__main__':
    freeze_support()
    main()  
