import os

OPT_SAVE_HYPEROPT = os.path.join(MODEL_PATH, 'opt_param_hyperopt_')


space_classic = {
    'learning_rate': hp.quniform('learning_rate', opt_fr[0], opt_fr[1], 0.01),
    'beta': hp.quniform('beta', opt_fr_beta[0], opt_fr_beta[1], 0.5)
}

space_best = {
    'lr3_train': hp.quniform('lr3_train', opt_fr[0], opt_fr[1], 0.01),
    'lr6_train': hp.quniform('lr6_train', opt_fr[0], opt_fr[1], 0.01),
    'lr3_test': hp.quniform('lr3_test', opt_fr[0], opt_fr[1], 0.01),
    'lr6_test': hp.quniform('lr6_test', opt_fr[0], opt_fr[1], 0.01),
    'beta': hp.quniform('beta', opt_fr_beta[0], opt_fr_beta[1], 0.5),
    'decay': hp.quniform('decay', opt_fr[0], opt_fr[1], 0.01),
    'pers': hp.quniform('pers', opt_fr[0], opt_fr[1], 0.01),
    'eps': hp.quniform('eps', opt_fr[0], opt_fr[1], 0.01),
    'init': hp.quniform('init', opt_fr[0], opt_fr[1], 0.01)
}


space_rlwm = {
    'learning_rate': hp.quniform('learning_rate', opt_fr[0], opt_fr[1], 0.01),
    'beta': hp.quniform('beta', opt_fr_beta[0], opt_fr_beta[1], 0.5),
    'decay': hp.quniform('decay', opt_fr[0], opt_fr[1], 0.01),
    'pers': hp.quniform('pers', opt_fr[0], opt_fr[1], 0.01),
    'eps': hp.quniform('eps', opt_fr[0], opt_fr[1], 0.01),
    'init': hp.quniform('init', opt_fr[0], opt_fr[1], 0.01),
    'eta3_wm': hp.quniform('eta3_wm', opt_fr[0], opt_fr[1], 0.01),
    'eta6_wm': hp.quniform('eta6_wm', opt_fr[0], opt_fr[1], 0.01)
}

space_rlwmi = {
    'learning_rate': hp.quniform('learning_rate', opt_fr[0], opt_fr[1], 0.01),
    'beta': hp.quniform('beta', opt_fr_beta[0], opt_fr_beta[1], 0.5),
    'decay': hp.quniform('decay', opt_fr[0], opt_fr[1], 0.01),
    'pers': hp.quniform('pers', opt_fr[0], opt_fr[1], 0.01),
    'eps': hp.quniform('eps', opt_fr[0], opt_fr[1], 0.01),
    'init': hp.quniform('init', opt_fr[0], opt_fr[1], 0.01),
    'eta3_wm': hp.quniform('eta3_wm', opt_fr[0], opt_fr[1], 0.01),
    'eta6_wm': hp.quniform('eta6_wm', opt_fr[0], opt_fr[1], 0.01)
}


# Generator of each cost function
def score_dict_gen(model_func, session):
  def opt_func(params):
    model = get_model(model_func, params)
    model.init_model(session.possible_stimuli, session.possible_actions)
    loss = cost_func_llh(model, session.train_set, session.test_set)
    #print(f'Params: {params}, Cost: {cost_total}')
    return {'loss': loss, 'status': STATUS_OK}
  return opt_func
  
  
# Hyperparameter-optimization: only run if needed
def search_solution_hopt(model_func, search_space, session, n_reps=opt_reps):

  score_func = score_dict_gen(model_func, session)
  trials_file = OPT_SAVE_HYPEROPT + model_func.__name__ + '_' + str(session.caseid) + '.trials'

  try:    
    trials = pickle.load(open(trials_file, 'rb'))
  except:
    trials = Trials()

  eval_step = 100
  eval_start = len(trials.trials) + eval_step

  if len(trials.trials) >= n_reps:
    eval_step = 1
    n_reps = len(trials.trials)
    eval_start = n_reps

  for tcount in range(eval_start, n_reps+1, eval_step):
    #print(f'Running cycle to {tcount}')
    best_param = fmin(score_func, search_space, 
                      algo=tpe.suggest, 
                      max_evals=tcount,
                      show_progressbar=False, 
                      trials = trials)
    pickle.dump(trials, open(trials_file, 'wb'))
    best_func = score_func(best_param)
  return best_param, best_func['loss']


def search_solution_all_hopt(model_func, search_space, session_list, n_reps):

  param_dict = {}
  funct_dict = {}
  for i in tqdm(range(len(session_list))):
    #print("**Optimizing session " + str(session_list[i].caseid))
    p, f = search_solution_hopt(model_func, search_space, session_list[i], n_reps) 
    param_dict[session_list[i].caseid] = p
    funct_dict[session_list[i].caseid] = f
  return param_dict, funct_dict


