import os
import sys
import random
from multiprocessing import Pool
import numpy as np
import scipy.optimize as optimize
import pickle
import inspect
from . import session, optimization
from .optimization import opt_fr, opt_fr_beta, cost_func_llh


OPT_SAVE_SCIPY = 'opt_param_scipy_'
OPT_SCIPY_EVALMAX = 1000


# Generator of each cost function
def opt_func_gen(model_func, session):
    def opt_func(params):
        model = model_func(*params)
        model.init_model(session.possible_stimuli, session.possible_actions)
        cost_total = cost_func_llh(model, session.train_set, session.test_set)
        #print(f'Params: {params}, Cost: {cost_total}')
        return cost_total
    return opt_func
  

# Search solution given constraints, repeating n_reps times
def search_solution_scipy(model_func, opt_bounds, session, n_reps, models_path):

    global OPT_SAVE_SCIPY
    global OPT_SCIPY_EVALMAX
  
    trials_file = OPT_SAVE_SCIPY + model_func.__name__ + '_' + str(session.caseid) + '.pickle'
    trials_file = os.path.join(models_path, trials_file)
    try:    
        trials = pickle.load(open(trials_file, 'rb'))
    except:
        trials = {}
        trials['params'] = np.empty(len(opt_bounds)).tolist()
        trials['min_val'] = sys.float_info.max
        trials['n_valid'] = 0
    n_reps = max(0, n_reps - trials['n_valid'])
    opt_func = opt_func_gen(model_func, session)
    #print(f"Optimizing case {session.caseid}")
    #pbar = tqdm(range(n_reps), position=0, leave=True)
    #for i in pbar:
    for i in range(n_reps): 
        x_init = [random.uniform(x0, x1) for x0, x1 in bounds_list]
        opt_res = optimize.minimize(opt_func, x_init,
                                    #method='SLSQP', #'TNC', 
                                    bounds=opt_bounds, 
                                    tol=1e-10,
                                    #options={'maxiter': OPT_SCIPY_EVALMAX}
                                    )
        if opt_res.success:
            if opt_res.fun < trials['min_val']:
                trials['min_val'] = opt_res.fun
                trials['params'] = opt_res.x.tolist()
            trials['n_valid'] += 1
            pickle.dump(trials, open(trials_file, "wb"))
        #pbar.set_description("Cost: %.2f Valid evals: %i" % (trials['min_val'], trials['n_valid']))
    return trials['min_val'], trials['params']


# Loop for optimization in a session list, using multiprocessing 
def search_solution_all_scipy(model_func, opt_bound, session_list, n_reps, models_path):
    param_dict = {}
    funct_dict = {}
    work_args = [(model_func, opt_bound, s, n_reps, models_path) for s in session_list]  
    #print(work_args)
    p = Pool(2)
    res = p.starmap(search_solution_scipy, work_args)
    #for i in tqdm(range(len(session_list)), position=0):
    #for i in range(len(session_list)):
    for i in range(len(res)):
        #print("**Optimizing session " + str(session_list[i].caseid))
        #f, p = search_solution_scipy(model_func, opt_bound, session_list[i], n_reps, models_path)
        f, p = res[i]
        param_dict[session_list[i].caseid] = p
        funct_dict[session_list[i].caseid] = f
    return param_dict, funct_dict
  

# Convert param list to dict
def param_list_to_dict(model_func, param_list):

    param_names = inspect.getfullargspec(model_func).args
    param_dict = {pname: v for pname, v in zip(param_names, param_list)}
    return param_dict


# Convert dict of params to ordered list
def param_dict_to_list(model_func, param_dict):

    param_names = inspect.getfullargspec(model_func).args
    param_list = [param_dict[pname for pname in param_names]
    return param_list




