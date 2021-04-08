import random
import numpy as np
from abc import ABC, abstractmethod
 

# Softmax helper function
def action_softmax(action_func, beta):
    scale = np.sum([np.exp(beta*f) for a, f in action_func.items()])
    sft_func = {a: np.exp(beta*f)/scale for a, f in action_func.items()} 
    return sft_func
        
# Base model
class BaseModel(ABC):
    
    @abstractmethod
    def init_model(self, stimuli, actions):
        pass

    @abstractmethod
    def learn_sample(self, stimulus, action, reward, block_size):
        pass
    
    @abstractmethod
    def get_policy(self, stimulus, block_size=None, test=False):
        pass

    def get_action(self, stimulus, block_size=None, test=False):
        pi = self.get_policy(stimulus, block_size, test)
        actions, probs = list(pi.keys()), list(pi.values())
        action = np.random.choice(actions, p=probs)
        return action


# Classic RL model class
class CollinsRLClassic(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta                    # softmax temperature
        self.__stmap = {}                   # Map of stimuli and respective actions
        self.__known_stimuli = set()        # stimuli already processed for init bias
        self.__Q = {}     
        self.__Q_init = 0.0     

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q[st] = {ac: self.__Q_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        st, ac, rt, bs = stimulus, action, reward, block_size
        self.__known_stimuli.add(st)
        # Delta calculation
        delta = rt - self.__Q[st][ac]
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + self.learning_rate*delta  

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus] 
        pi_rl = action_softmax(Q_st, self.beta)
        return pi_rl


# Generic model class
class CollinsRLBest(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta):
        self.lr3_train = learning_rate
        self.lr6_train = learning_rate
        self.lr3_test = learning_rate
        self.lr6_test = learning_rate
        self.beta = beta                            # softmax temperature
        self.eps = 0.0                              # noise ratio
        self.phi = 0.0                              # forgetting ratio / decay
        self.pers = 0.0                             # perseveration param
        self.init = 0.0                             # init bias param
        self.__stmap = {}                           # Map of stimuli and respective actions
        self.__known_stimuli = set()                # stimuli already processed for init bias
        self.__Q_train = {}      
        self.__Q_test = {}       
        self.__Q_init = 0.0     

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q_train[st] = {ac: self.__Q_init for ac in actions}
            self.__Q_test[st] = {ac: self.__Q_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        lr_train = self.lr3_train if block_size == 3 else self.lr6_train
        lr_test = self.lr3_test if block_size == 3 else self.lr6_test
        # Forgetting - fix to case with different Q/W
        for s, actions in self.__stmap.items():
            for a in actions:
                self.__Q_train[s][a] = (1.-self.phi)*self.__Q_train[s][a] + self.phi*self.__Q_init
                self.__Q_test[s][a] = (1.-self.phi)*self.__Q_test[s][a] + self.phi*self.__Q_init
        # Initial bias update  
        if st not in self.__known_stimuli:
            self.__Q_train[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            #self.__Q_test[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Delta calculation
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

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q_test[stimulus] if test else self.__Q_train[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl)
        pi = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        return pi


# RLWMi model class
class CollinsRLWM(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__Q = {}     
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
            self.__Q[st] = {ac: self.__Q_init for ac in actions}
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        # Forgetting - fix to case with different Q/W
        if self.phi > 0:
            for s, actions in self.__stmap.items():
                for a in actions:
                    self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                    self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Initial bias update  
        if st not in self.__known_stimuli:
            self.__Q[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if rt < 1.0: #delta < 0:
            lr = lr*(1. - self.pers)
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + lr*delta  
        self.__W[st][ac] = rt  

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus]
        W_st = self.__W[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl.keys())
        pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_rl
        pi = {}
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi


# RLWM model class version 2
class CollinsRLWMAlt(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta, K, coupled=False):
        self.alpha_rl = learning_rate
        self.alpha_wm = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.eta_wm = 0.0                       # wm weight in policy calculation
        self.K = K                              # WM capacity
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__Q = {}     
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
            self.__Q[st] = {ac: self.__Q_init for ac in actions}
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        # Forgetting 
        for s, actions in self.__stmap.items():
            for a in actions:
                #self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Delta calculation
        if self.coupled:
            delta_rl = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta_rl = rt - self.__Q[st][ac]
        delta_wm = rt - self.__W[st][ac]
        # Perseveration
        alpha_rl = self.alpha_rl
        alpha_wm = self.alpha_wm
        if rt < 1.:
            alpha_rl = alpha_rl*(1. - self.pers)
            alpha_wm = alpha_wm*(1. - self.pers)
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + alpha_rl*delta_rl  
        self.__W[st][ac] = self.__W[st][ac] + alpha_wm*delta_wm 

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus]
        W_st = self.__W[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl.keys())
        pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_rl
        pi = {}
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi


# Model: Classic RL
def model_classic(learning_rate, beta):
    model = CollinsRLClassic(learning_rate, beta)
    return model


# Model: Best RL with improvements
def model_best(lr3_train, lr6_train, lr3_test, lr6_test, beta, decay, pers, eps, init):
    model = CollinsRLBest()
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
    
    
# Model: non-interacting RL+WM  
def model_rlwm(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = CollinsRLWM(learning_rate, beta, coupled=False)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model
    
    
# Model: interacting RL+WM
def model_rlwmi(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = CollinsRLWM(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model


# Model: RLWM alternative version
def model_rlwma(alpha_rl, alpha_wm, beta, decay, pers, eps, eta_wm, K):
    model = CollinsRLWMAlt(alpha_rl, beta, K, coupled=False)
    model.alpha_rl = alpha_rl
    model.alpha_wm = alpha_wm
    model.beta = beta
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.eta_wm = eta_wm
    model.K = K
    return model
     

# Returns a model expanding parameters into arguments
def get_model(model_func, params):
    #print(model_func.__name__, type(params), params)
    model = None
    if isinstance(params, dict):
        model = model_func(**params)
    if isinstance(params, list):
        model = model_func(*params)
    return model
