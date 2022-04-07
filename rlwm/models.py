import random
import numpy as np
from abc import ABC, abstractmethod
 

# Softmax helper function
def action_softmax(action_func, beta):
    scale = np.sum([np.exp(beta*f) for a, f in action_func.items()])
    sft_func = {a: np.exp(beta*f)/scale for a, f in action_func.items()} 
    return sft_func
        
# Helper function to extract stimulus prefix
def get_stimulus_group(stimulus):
    return stimulus[:-1]

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


# Collins RLWMi model class
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
        if delta < 0.:
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


# RLWM model class based on most recent source 
class CollinsRLWMalt1(BaseModel):
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
        #print(delta_rl, alpha_rl, round(alpha_rl*delta_rl, 3))
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


# RLWM model merge of rlwmi and alt1 
class CollinsRLWMalt2(BaseModel):
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
                    #self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
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
        if rt < 1.0:
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


# WM only model 
class CollinsWM(BaseModel):
    '''WM only model with additional mechanisms'''  
    def __init__(self, alpha_wm, beta, K):
        self.alpha_wm = alpha_wm
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.eta_wm = 0.0                       # wm weight in policy calculation
        self.K = K                              # WM capacity
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__W = {}
        self.__W_init = 0.0

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__W_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        # Forgetting 
        for s, actions in self.__stmap.items():
            for a in actions:
                self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        delta_wm = rt - self.__W[st][ac]
        # Perseveration
        alpha_wm = self.alpha_wm
        if rt < 1.:
            alpha_wm = alpha_wm*(1. - self.pers)
        # Function updates
        self.__W[st][ac] = self.__W[st][ac] + alpha_wm*delta_wm 

    def get_policy(self, stimulus, block_size=None, test=False):
        W_st = self.__W[stimulus]
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_wm.keys())
        pi_other = {ac: 1/n_a for ac in pi_wm.keys()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_other
        pi = {}
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        for ac in pi_wm.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_other[ac]
        return pi

# Collins RLWMi + deduction version 1
class RLWMnew1(BaseModel):
    '''RLWM model with deduction of non random answers'''  
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
        self.__known_answers = {}
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
        # Rewarding deduction/induction 
        if st not in self.__known_answers and rt > 0:
            st_group = get_stimulus_group(st)
            st_coef = 1. - (3./block_size)
            for s, actions in self.__stmap.items():
                if s != st and get_stimulus_group(s) == st_group:
                    # Change here to affect only one learning mechanism
                    self.__Q[s][ac] = self.__Q[s][ac]*st_coef
                    self.__W[s][ac] = self.__W[s][ac]*st_coef
            self.__known_answers[st] = ac        
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
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


# Collins RLWMi + deduction version 2
class RLWMnew2(BaseModel):
    '''RLWM model with deduction of non random answers'''  
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.gamma_rl = 1.0
        self.gamma_wm = 1.0
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__known_answers = {}
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
        # Rewarding deduction/induction 
        if st not in self.__known_answers and rt > 0:
            st_group = get_stimulus_group(st)
            for s, actions in self.__stmap.items():
                if s != st and get_stimulus_group(s) == st_group:
                    if s not in self.__known_answers or self.__known_answers[s] != ac:
                        # Change here to affect only one learning mechanism
                        self.__Q[s][ac] = self.__Q[s][ac]*self.gamma_rl
                        self.__W[s][ac] = self.__W[s][ac]*self.gamma_wm
            self.__known_answers[st] = ac        
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
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


# Collins RLWMi + deduction version 3
class RLWMnew3(BaseModel):
    '''RLWM model with deduction of non random answers'''  
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.gamma_pos = 0.0
        self.gamma_neg = 0.0
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
        # Rewarding deduction/induction 
        st_group = get_stimulus_group(st)
        for s, actions in self.__stmap.items():
            if s != st and get_stimulus_group(st) == st_group:
                if rt > 0:
                    self.__Q[s][ac] = self.__Q[s][ac]*(1. - self.gamma_pos*3./block_size)
                    self.__W[s][ac] = self.__W[s][ac]*(1. - self.gamma_pos*3./block_size)
                else:
                    self.__Q[s][ac] = self.__Q[s][ac]*(1. + self.gamma_neg*3./block_size)
                    self.__W[s][ac] = self.__W[s][ac]*(1. + self.gamma_neg*3./block_size)
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
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


# Model: Classic RL
def model_classic(learning_rate, beta):
    model = CollinsRLClassic(learning_rate, beta)
    return model


# Model: Best RL with improvements
def model_best(lr3_train, lr6_train, lr3_test, lr6_test, beta, decay, pers, eps, init):
    model = CollinsRLBest(max(lr3_train, lr6_train), beta)
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
    model = CollinsRLWMalt1(alpha_rl, beta, K, coupled=False)
    model.alpha_rl = alpha_rl
    model.alpha_wm = alpha_wm
    model.beta = beta
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.eta_wm = eta_wm
    model.K = K
    return model
     

# Model: merge of RLWM and alt1
def model_rlwmb(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = CollinsRLWMalt2(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model

# Model: WM only model
def model_wm(alpha_wm, beta, decay, pers, eps, eta_wm, K):
    model = CollinsWM(alpha_wm, beta, K)
    model.alpha_wm = alpha_wm
    model.beta = beta
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.eta_wm = eta_wm
    model.K = K
    return model

# Model: interacting RL+WM version 1
def model_new1(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = RLWMnew1(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model

# Model: interacting RL+WM version 2
def model_new2(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm, gamma_rl, gamma_wm):
    model = RLWMnew2(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    model.gamma_rl = gamma_rl
    model.gamma_wm = gamma_wm
    return model


## Model: interacting RL+WM version 3
def model_new3(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm, gamma_pos, gamma_neg):
    model = RLWMnew3(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    model.gamma_pos = gamma_pos
    model.gamma_neg = gamma_neg
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
