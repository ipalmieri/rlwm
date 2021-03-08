import numpy as np

# optmization parameters
opt_fr = (0., 1.)
opt_fr_beta = (0, 100)
opt_reps = 20
opt_evalmax= 1000


# LLH estimation - function to be MINIMIZED
def cost_func_llh(model, train_set, test_set):

  prob_seq_train = []
  prob_seq_test = []

  for trial in train_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=False)
    ac_prob = pi[ac]
    prob_seq_train.append(ac_prob)
    model.learn_sample(st, ac, rt, bs)

  for trial in test_set:
    st, ac, rt, bs = trial
    pi = model.get_policy(st, bs, test=True)
    ac_prob = pi[ac]
    prob_seq_test.append(ac_prob)

  llh_train = - np.sum(np.log(prob_seq_train))
  llh_test = - np.sum(np.log(prob_seq_test))

  return llh_train + llh_test



def save_param_dict(filename, param_dict, func_dict=None):
  df_param = pd.DataFrame.from_dict(param_dict, orient='index')
  if func_dict:
    df_func = pd.DataFrame.from_dict(func_dict, orient='index', columns=['min_cost'])
  df_param = df_param.merge(df_func, how='left', left_index=True, right_index=True)
  #print(df_param)
  df_param.to_csv(filename, sep=';', index_label='caseid')
  df_param.to_excel(filename, index_label='caseid')
