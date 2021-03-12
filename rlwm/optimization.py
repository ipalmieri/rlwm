import numpy as np
import pandas as pd
#from multiprocessing import Pool


# LLH estimation - function to be MINIMIZED
def cost_func_llh(model, session):

  prob_seq_train = []
  prob_seq_test = []

  for trial in session.train_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=False)
    ac_prob = pi[ac]
    prob_seq_train.append(ac_prob)
    model.learn_sample(st, ac, rt, bs)

  for trial in session.test_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=True)
    ac_prob = pi[ac]
    prob_seq_test.append(ac_prob)

  llh_train = - np.sum(np.log(prob_seq_train))
  llh_test = - np.sum(np.log(prob_seq_test))

  return llh_train + llh_test


# Save the dict param into spreasheets
def save_param_dict(filename, param_dict, func_dict=None):
  df_param = pd.DataFrame.from_dict(param_dict, orient='index')
  if func_dict:
    df_func = pd.DataFrame.from_dict(func_dict, orient='index', columns=['min_cost'])
  df_param = df_param.merge(df_func, how='left', left_index=True, right_index=True)
  if not (filename.endswith('.xls') or filename.endswith(".xlsx")):
      filename = filename + str('.xlsx')
  df_param.to_excel(filename, index_label='caseid')
 

# Loop for optimization in a session list, using multiprocessing 
def search_sessions_solution(opt_routine, model_func, opt_bounds, session_list, n_reps, models_path, n_jobs=None):
    param_dict = {}
    funct_dict = {}
    #work_args = [(model_func, opt_bounds, s, n_reps, models_path) for s in session_list]
    #n_jobs = None if n_jobs <= 0 else n_jobs
    #p = Pool(processes=n_jobs)
    #res = p.starmap(opt_routine, work_args)
    #p.close()
    #p.join()
    #for i in tqdm(range(len(session_list)), position=0):
    #for i in range(len(session_list)):
    for i in range(len(res)):
        #print("**Optimizing session " + str(session_list[i].caseid))
        p, f = opt_routine(model_func, opt_bound, session_list[i], n_reps, models_path)
        #p, f = res[i]
        param_dict[session_list[i].caseid] = p
        funct_dict[session_list[i].caseid] = f
    return param_dict, funct_dict

