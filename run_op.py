import os
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.models as models
import rlwm.optsp as optsp
import rlwm.optimization as optimization

BASE_PATH = '/srv/black/data/rlwm'
DATA_PATH = os.path.join(BASE_PATH, 'dados')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
MODEL_PATH = os.path.join(BASE_PATH, 'models')
OPT_REPS = 20


CASEIDS = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 17, 25, 26, 27, 29, 37, 49, 54, 57, 59, 62, 64, 66, 72, 76, 77, 79, 84, 91, 92, 94, 97, 102, 105, 110, 118, 127, 132, 153, 155, 159, 161, 164, 166, 172, 174, 175, 178, 179, 180, 181, 182, 184, 185, 187, 189, 195, 197, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 214, 215, 216, 217, 218, 220, 221, 223, 224, 226, 227]




BOUNDS_CLASSIC = {'learning_rate':  (0., 1.),
                  'beta':           (0., 100)}
                  



def main():

    # Load all datasets
    session_list = []
    for id in CASEIDS:
      ds = session.load_session(id, DATA_PATH)
      session_list.append(ds)
    print(f'{len(session_list)} cases loaded')
 


    p_sp_classic, f_sp_classic = optsp.search_solution_all(models.model_classic, 
                                                           BOUNDS_CLASSIC, 
                                                           session_list, 
                                                           n_reps=OPT_REPS, 
                                                           models_path=MODEL_PATH,
                                                           n_jobs=2
                                                           )
    optimization.save_param_dict(os.path.join(OUTPUT_PATH, 'param_sp_classic'), p_sp_classic, f_sp_classic)


if __name__ == '__main__':
    freeze_support()
    main()  
