import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from . import optsp, optho

OPT_DEFAULT_SOLVER='scipy'

# LLH estimation - function to be MINIMIZED
# Includes only train phase data
def cost_half_llh(model, session):

  prob_seq_train = []
  for trial in session.train_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=False)
    ac_prob = pi[ac]
    prob_seq_train.append(ac_prob)
    model.learn_sample(st, ac, rt, bs)
  llh_train = np.sum(np.log(prob_seq_train))
  cost = - llh_train 
  return cost


# LLH estimation - function to be MINIMIZED
# Includes train + test phases
def cost_full_llh(model, session):

  prob_seq_train = []
  prob_seq_test = []

  for trial in session.train_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=False)
    ac_prob = pi[ac]
    prob_seq_train.append(ac_prob)
    model.learn_sample(st, ac, rt, bs)
    #ac_i = model.get_action(st, bs, test=False)
    #rt_i = session.get_reward(st, ac_i)
    #model.learn_sample(st, ac_i, rt_i, bs)

  for trial in session.test_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=True)
    ac_prob = pi[ac]
    prob_seq_test.append(ac_prob)

  llh_train = np.sum(np.log(prob_seq_train))
  llh_test = np.sum(np.log(prob_seq_test))
  cost = - (llh_train + llh_test)
  return cost


# Save the dict param into spreasheets
def save_param_dict(filename, param_dict, func_dict=None):
  df_param = pd.DataFrame.from_dict(param_dict, orient='index')
  if func_dict:
    df_func = pd.DataFrame.from_dict(func_dict, orient='index', columns=['min_cost'])
  df_param = df_param.merge(df_func, how='left', left_index=True, right_index=True)
  k = len(df_param.columns) - 1 # cols except caseid and min_cost
  # AIC = 2*k - 2*np.log(LLH) 
  # min_cost = -np.log(LLH)
  df_param['aic'] = 2.*k + 2.*df_param['min_cost']
  if not (filename.endswith('.xls') or filename.endswith(".xlsx")):
      filename = filename + str('.xlsx')
  df_param.to_excel(filename, index_label='caseid')


# Builds model name from optimization attributes
def get_model_name(model_func, session, solver=OPT_DEFAULT_SOLVER, model_name=None):
    name = 'opt_param'
    name = name + '_' + solver + '_'
    if model_name:
        name = name + model_name
    else:
        name = name + model_func.__name__
    name = name + '_' + str(session.caseid)
    name = name + '.pickle'
    return name


# Returns optimization routine for a given solver name
def _get_solver_module(solver):
    if solver == 'scipy':
        return optsp
    if solver == 'hyperopt':
        return optho
    return None


# Returns model params from model file
def get_model_params(model_func, session, solver=OPT_DEFAULT_SOLVER, model_name=None, model_path=None):
    model_file = get_model_name(model_func, session, solver, model_name)
    if model_path:
        model_file = os.path.join(model_path, model_file)
    params, loss = _get_solver_module(solver).load_params(model_file)
    return params, loss


# Loop for optimization in a session list, using multiprocessing 
def search_solution_mp(model_func, opt_bounds, session_list, n_reps, solver=OPT_DEFAULT_SOLVER, models_path = None, model_name=None, n_jobs=None):
    param_dict = {}
    funct_dict = {}
    opt_routine = _get_solver_module(solver).search_solution
    work_args = []
    for session in session_list:
        filename = os.path.join(models_path, get_model_name(model_func, session, solver, model_name)) if models_path else None
        args = (model_func, opt_bounds, session, n_reps, filename)
        work_args.append(args)
    n_jobs = None if n_jobs <= 0 else n_jobs
    p = Pool(processes=n_jobs)
    res = p.starmap(opt_routine, work_args)
    p.close()
    p.join()
    for i in range(len(session_list)):
        #p, f = opt_routine(model_func, opt_bounds, session_list[i], n_reps, models_path)
        p, f = res[i]
        param_dict[session_list[i].caseid] = p
        funct_dict[session_list[i].caseid] = f
    return param_dict, funct_dict

