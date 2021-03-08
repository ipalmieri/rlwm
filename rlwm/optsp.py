import inspect


# Definition of initial and bound values
bounds_classic = [opt_fr, opt_fr_beta] 
bounds_best = [opt_fr, opt_fr, opt_fr, opt_fr, opt_fr_beta, opt_fr, opt_fr, opt_fr, opt_fr]
bounds_rlwm = [opt_fr, opt_fr_beta, opt_fr, opt_fr, opt_fr, opt_fr, opt_fr, opt_fr]
bounds_rlwmi = bounds_rlwm    

p_sp_classic, p_sp_classic_l, f_sp_classic = {}, {}, {}
p_sp_best, p_sp_best_l, f_sp_best = {}, {}, {}
p_sp_rlwm, p_sp_rlwm_l, f_sp_rlwm = {}, {}, {}
p_sp_rlwmi, p_sp_rlwmi_l, f_sp_rlwmi = {}, {}, {}



# Generator of each cost function
def opt_func_gen(model_func, session):
  def opt_func(params):
    model = model_func(*params)
    model.init_model(session.possible_stimuli, session.possible_actions)
    cost_total = cost_func_llh(model, session.train_set, session.test_set)
    #print(f'Params: {params}, Cost: {cost_total}')
    return cost_total
  return opt_func
  
  
  
# Optimization routines

def param_init_gen(bounds_list):
    return [random.uniform(x0, x1) for x0, x1 in bounds_list]


def search_solution_scipy(model_func, opt_bounds, session, n_reps=opt_reps):

  trials_file = OPT_SAVE_SCIPY + model_func.__name__ + '_' + str(session.caseid) + '.pickle'
  try:    
    trials = pickle.load(open(trials_file, 'rb'))
  except:
    trials = {}
    trials['params'] = np.empty(len(opt_bounds)).tolist()
    trials['min_val'] = sys.float_info.max
    trials['n_valid'] = 0

  n_reps = max(0, n_reps - trials['n_valid'])
  opt_func = opt_func_gen(model_func, session)
  #pbar = tqdm(range(n_reps), position=0, leave=True)
  #for i in pbar:
  for i in range(n_reps): 
    x_init = param_init_gen(opt_bounds)
    opt_res = optimize.minimize(opt_func, x_init,
                                #method='SLSQP', #'TNC', 
                                bounds=opt_bounds, 
                                tol=1e-10,
                                options={'maxiter': opt_evalmax})
    if opt_res.success:
      if opt_res.fun < trials['min_val']:
        trials['min_val'] = opt_res.fun
        trials['params'] = opt_res.x.tolist()
      trials['n_valid'] += 1
      pickle.dump(trials, open(trials_file, "wb"))
      #pbar.set_description("Cost: %.2f Valid evals: %i" % (trials['min_val'], trials['n_valid']))
  return trials['min_val'], trials['params']


def search_solution_all_scipy(model_func, opt_bound, session_list, n_reps):
  param_dict = {}
  funct_dict = {}
  #for i in tqdm(range(len(session_list)), position=0):
  for i in range(len(session_list)):
    print("**Optimizing session " + str(session_list[i].caseid))
    f, p = search_solution_scipy(model_func, opt_bound, session_list[i], n_reps)
    param_dict[session_list[i].caseid] = p
    funct_dict[session_list[i].caseid] = f
  return param_dict, funct_dict
  
  


def param_list_dict(model_func, case_dict):

  ret_dict = {}
  param_names = inspect.getfullargspec(model_func).args
  for caseid, param_list in case_dict.items():
    ret_dict[caseid] = {n: v for n, v in zip(param_names, param_list)}
  return ret_dict
