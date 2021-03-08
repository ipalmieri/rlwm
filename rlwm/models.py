import random
import numpy as np




def action_softmax(action_func, beta):
  scale = 1./np.sum([np.exp(beta*f) for a, f in action_func.items()])
  return {a: np.exp(beta*f)*scale for a, f in action_func.items()} 
    
    
    
class RLWMCollins():
  '''RL model with additional mechanisms'''  
  def __init__(self, learning_rate=0.1, beta=1.0):
    self.lr3_train = learning_rate
    self.lr6_train = learning_rate
    self.lr3_test = learning_rate
    self.lr6_test = learning_rate
    self.beta = beta              # softmax temperature
    self.eps = 0.0                # noise ratio
    self.phi = 0.0                # forgetting ratio / decay
    self.pers = 0.0               # perseveration param
    self.init = 0.0               # init bias param
    self.eta3_wm = 0.0            # wm weight in policy calculation
    self.eta6_wm = 0.0            # wm weight in policy calculation
    self.coupled = False          # True for RL + WM interacting model
    self.__stmap = {}             # Map of stimuli and respective actions
    self.__known_stimuli = set()  # stimuli already processed for init bias
    self.__Q_train = {}   
    self.__Q_test = {}     
    self.__W = {}
    self.__Q_init = 0.0   
    self.__W_init = 0.0

  def init_model(self, stimuli, actions):
    actions = set(actions)
    stimuli = set(stimuli)
    self.__stmap = {st: actions for st in stimuli}
    self.__Q_init = 1./len(actions) # alternative: 0 
    self.__W_init = 1./len(actions) # alternative: 0 
    for st in stimuli:
      self.__Q_train[st] = {ac: self.__Q_init for ac in actions}
      self.__Q_test[st] = {ac: self.__Q_init for ac in actions}
      self.__W[st] = {ac: self.__W_init for ac in actions}

  def learn_sample(self, stimulus, action, reward, block_size):
    #print(sample, block_size)
    st, ac, rt = stimulus, action, reward
    # Block size dependent parameters
    lr_train = self.lr3_train if block_size == 3 else self.lr6_train
    lr_test = self.lr3_test if block_size == 3 else self.lr6_test
    eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
    # Forgetting - fix to case with different Q/W
    if self.phi > 0:
      for s, actions in self.__stmap.items():
        for a in actions:
          self.__Q_train[s][a] = (1.-self.phi)*self.__Q_train[s][a] + self.phi*self.__Q_init
          self.__Q_test[s][a] = (1.-self.phi)*self.__Q_test[s][a] + self.phi*self.__Q_init
          self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
    # Initial bias update  
    if st not in self.__known_stimuli:
      self.__Q_train[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
      #self.__Q_test[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
      self.__known_stimuli.add(st)
    # Delta calculation
    if self.coupled:
      delta_train = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q_train[st][ac])
      delta_test = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q_test[st][ac])
    else:
      delta_train = rt - self.__Q_train[st][ac]
      delta_test = rt - self.__Q_test[st][ac]
    # Perseveration
    if delta_train < 0:
      lr_train = lr_train*(1. - self.pers)
    if delta_test < 0:  
      lr_test = lr_test*(1. - self.pers)
    # Function updates
    self.__Q_train[st][ac] = self.__Q_train[st][ac] + lr_train*delta_train  
    self.__Q_test[st][ac] = self.__Q_test[st][ac] + lr_test*delta_test  
    self.__W[st][ac] = rt  

  def get_policy(self, stimulus, block_size=None, test=False):
    Q_st = self.__Q_test[stimulus] if test else self.__Q_train[stimulus]
    W_st = self.__W[stimulus]
    pi_rl = action_softmax(Q_st, self.beta)
    pi_wm = action_softmax(W_st, self.beta)
    # Undirected noise
    if self.eps > 0:
      n_a = len(pi_rl)
      pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
      pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
    # Final policy - mixed WM and RL
    if block_size and not test:
      pi = {}
      eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
      for ac in Q_st.keys():
        pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
      return pi
    return pi_rl

  def get_action(self, stimulus, block_size=None, test=False):
    pi = self.get_policy(stimulus, block_size, test)
    actions, probs = list(pi.keys()), list(pi.values())
    action = np.random.choice(actions, p=probs)
    return action




def model_classic(learning_rate, beta):
  model = RLWMCollins(learning_rate, beta)
  return model




def model_best(lr3_train, lr6_train, lr3_test, lr6_test, beta, decay, pers, eps, init):
  model = RLWMCollins()
  model.lr3_train = lr3_train
  model.lr6_train = lr6_train
  model.lr3_test = lr3_test
  model.lr6_test = lr6_test
  model.beta = beta
  model.phi = decay
  model.pers = pers
  model.eps = eps
  model.init = init
  return model
  
  
  
def model_rlwm(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
  model = RLWMCollins(learning_rate, beta)
  model.phi = decay
  model.pers = pers
  model.eps = eps
  model.init = init
  model.eta3_wm = eta3_wm
  model.eta6_wm = eta6_wm
  return model
  
  
 

def model_rlwmi(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
  model = RLWMCollins(learning_rate, beta)
  model.phi = decay
  model.pers = pers
  model.eps = eps
  model.init = init
  model.eta3_wm = eta3_wm
  model.eta6_wm = eta6_wm
  model.coupled = True
  return model




def get_model(model_func, params):
  model = None
  if isinstance(params, dict):
    model = model_func(**params)
  if isinstance(params, list):
    model = model_func(*params)
  return model
