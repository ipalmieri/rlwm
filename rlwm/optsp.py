import os
import sys
import random
import numpy as np
import pickle
import inspect
import scipy.optimize as optimize
from . import session, optimization, models


OPT_SAVE_SCIPY = 'opt_param_scipy_'
OPT_SCIPY_EVALMAX = 1000
OPT_TOL = 1e-10

# Generator of each cost function
def generate_optfunc(model_func, session):
    def opt_func(params):
        params = params.tolist()
        model = models.get_model(model_func, params)
        model.init_model(session.possible_stimuli, session.possible_actions)
        cost_total = optimization.cost_func_llh(model, session)
        #print(f'Params: {params}, Cost: {cost_total}')
        return cost_total
    return opt_func
  

# Search solution given constraints, repeating n_reps times
def search_solution(model_func, opt_bounds, session, n_reps, models_path):

    global OPT_SAVE_SCIPY
    global OPT_SCIPY_EVALMAX
    global OPT_TOL
 
    # Try to load file with past runs
    trials_file = OPT_SAVE_SCIPY + model_func.__name__ + '_' + str(session.caseid) + '.pickle'
    trials_file = os.path.join(models_path, trials_file)
    try:    
        trials = pickle.load(open(trials_file, 'rb'))
    except:
        trials = {}
        trials['params'] = {}
        trials['min_val'] = sys.float_info.max
        trials['n_valid'] = 0
    n_reps = max(0, n_reps - trials['n_valid'])
    # Define function and bounds for minimize()
    opt_func = generate_optfunc(model_func, session)
    param_names = inspect.getfullargspec(model_func).args
    bounds_list = [opt_bounds[pname] for pname in param_names]
    # Optmization loop
    print(f"Optimizing case {session.caseid}")
    #pbar = tqdm(range(n_reps), position=0, leave=True)
    #for i in pbar:
    for i in range(n_reps): 
        x_init = [random.uniform(x0, x1) for x0, x1 in bounds_list]
        opt_res = optimize.minimize(opt_func, x_init,
                                    #args=(model_func, session),
                                    #method='TNC', #'Powell', #'SLSQP', #'TNC', 
                                    bounds=bounds_list, 
                                    tol=OPT_TOL,
                                    #options={'maxiter': OPT_SCIPY_EVALMAX}
                                    )
        if opt_res.success:
            if opt_res.fun < trials['min_val']:
                trials['min_val'] = opt_res.fun
                trials['params'] = {pname: v for pname, v in zip(param_names, opt_res.x.tolist())}
            trials['n_valid'] += 1
            pickle.dump(trials, open(trials_file, "wb"))
        #pbar.set_description("Cost: %.2f Valid evals: %i" % (trials['min_val'], trials['n_valid']))
    return trials['params'], trials['min_val']


# Loop for optimization in a session list, using multiprocessing 
def search_solution_all(model_func, opt_bounds, session_list, n_reps, models_path, n_jobs):
    ret = optimization.search_sessions_solution(search_solution, 
                                                model_func, 
                                                opt_bounds, 
                                                session_list, 
                                                n_reps, 
                                                models_path,
                                                n_jobs)
    return ret
