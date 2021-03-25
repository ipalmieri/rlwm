import os
from multiprocessing import freeze_support
import rlwm.models as models
import rlwm.session as session
import rlwm.transformation as transformation
import rlwm.simulation as simulation
import rlwm.optsp as optsp
import rlwm.optho as optho


BASE_PATH = '/srv/black/data/rlwm'
DATA_PATH = os.path.join(BASE_PATH, 'dados')
#OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
#MODEL_PATH = os.path.join(BASE_PATH, 'models')
OUTPUT_PATH = '/tmp/rlwm'
MODEL_PATH = '/tmp/rlwm'
                  



def main():

    file_set = set()
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".csv"):
             file_set.add(int(filename.split('-')[0]))
    print(list(file_set))


    CASEIDS = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 17, 25, 26, 27, 29, 
               37, 49, 54, 57, 59, 62, 64, 66, 72, 76, 77, 79, 84, 
               91, 92, 94, 97, 102, 105, 110, 118, 127, 132, 153, 155, 
               159, 161, 164]

    session_list = []

    for id in CASEIDS:
        ds = session.load_session(id, DATA_PATH)
        session_list.append(ds)
    print(f'{len(session_list)} cases loaded')


    opt_param_classic = {'learning_rate': 0.3, 
                         'beta':            8
                        }
    bounds_classic = {'learning_rate':  (0., 1.),
                      'beta':           (0., 100)
                     }

    opt_param_rlwmi = {'learning_rate': 0.3,
                       'beta':          8.0,
                       'decay':         0.1,
                       'pers':          0.1,
                       'eps':           0.2,
                       'init':          0.5,
                       'eta3_wm':       0.6,
                       'eta6_wm':       0.3
                      }

    bounds_rlwmi = {'learning_rate':    (0., 1.),
                       'beta':          (0., 10),
                       'decay':         (0., 1.),
                       'pers':          (0., 1.),
                       'eps':           (0., 1.),
                       'init':          (0., 1.),
                       'eta3_wm':       (0., 1.),
                       'eta6_wm':       (0., 1.)
                      }


    opt_reps = 20
    opt_evalmax = 1000
 
    opt_params = opt_param_rlwmi
    opt_bounds = bounds_rlwmi
    opt_modelfunc = models.model_rlwmi


    opt_session = session_list[2]

#    opt_model = models.get_model(opt_modelfunc, opt_params)
#    opt_model.init_model(opt_session.possible_stimuli, opt_session.possible_actions)
#    model_session = simulation.simulate_model_session(opt_model, opt_session)


#    opt_session_list = [model_session]
#    opt_session_list = [opt_session]
    opt_session_list = session_list

#    p_sp, f_sp = optsp.search_solution_all(model_func=opt_modelfunc,
#                                           opt_bounds=opt_bounds,
#                                           session_list=opt_session_list,
#                                           n_reps=opt_reps,
#                                           models_path=None,
#                                           n_jobs=2
#                                          )
#
#
#    print(p_sp, f_sp)



    p_ho, f_ho = optho.search_solution_all(model_func=opt_modelfunc, 
                                           opt_bounds=opt_bounds, 
                                           session_list=opt_session_list, 
                                           n_reps=opt_evalmax, 
                                           models_path=None,
                                           n_jobs=2
                                          )

    print(p_ho, f_ho)





if __name__ == '__main__':
    freeze_support()
    main()  
