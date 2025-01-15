from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import os
from tqdm import tqdm
from xgboost import XGBRegressor
import zstandard as zstd
import pickle
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

parameters = {"horizon": 600000, "max_episode": 150, "gamma": 0.97, "n_estimators": 650, "max_depth": 15}

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

# I tried many configurations using DQN and FIQ (based on the work I did for class number 4). I played a lot with hypermarameters doing a lot of grid search and optimization.
# The best results I got are with the following configuration (FIQ algorithm), with DQN or DDQN I neved managed to pass the third test. (I just put here the most promising
# not to be too heavy).
# The code should run well. In any case, don't hesitate to contact me to get more information or to help you to run the code.
# I also put the learning rate to 0.05 (see below). I used XGBRegressor rather than RandomForestRegressor.
 
class ProjectAgent:
    def __init__(self, config=parameters, env=env):
        
        #Hyperparameters
        self.horizon = config["horizon"]
        self.max_episode = config["max_episode"]
        self.gamma = config["gamma"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]

        #Defintition of model and environment
        self.env = env
        self.model = None
    
    def collect_samples(self, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = self.env.reset()
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2

        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        
        return S, A, R, S2, D
    

    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        #I initialize the scores to 0 
        indiv_test = 0
        pop_test = 0

        nb_samples = S.shape[0]
        Qfunctions = []

        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2

            Q = XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate = 0.05)
            Q.fit(SA,value)
            self.model = Q
            Qfunctions.append(Q)

                
            if iter % 5 == 0:
                print(f" Iteration {iter}: Evaluating the model .......")
                current_indiv_score = evaluate_HIV(agent=self, nb_episode=5)
                current_pop_score = evaluate_HIV_population(agent=self, nb_episode=20)
                print(f"Scores are for individual: {current_indiv_score}, population: {current_pop_score}")

                #I check the new scores and update the best model
                if current_indiv_score > indiv_test:
                    indiv_test = current_indiv_score
                    self.save()
                    print("score better for indiv")

                elif current_indiv_score == indiv_test and current_pop_score > pop_test:
                    pop_test = current_pop_score
                    self.save()
                    print("score better for pop")


        return Qfunctions
    
    def train(self, disable_tqdm=False, print_done_states=False):
        #This is the train function
        S, A, R, S2, D = self.collect_samples(self.horizon, disable_tqdm, print_done_states)
        nb_actions = self.env.action_space.n
        self.Qfunctions = self.rf_fqi(S, A, R, S2, D, self.max_episode, nb_actions, self.gamma, disable_tqdm)
        self.model = self.Qfunctions[-1]
        return self.model

    def greedy_action(self, Q,s,nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)

    # def greedy_action(self, Q, s, nb_actions):
    #     Qsa = []
    #     for a in range(nb_actions):
    #         sa = np.hstack((s.reshape(1, -1), [[a]])).reshape(1, -1)
    #         Qsa.append(Q.predict(sa).item())
    #     return np.argmax(Qsa)
    
    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        else:
            return self.greedy_action(self.model, observation, self.env.action_space.n)

#I save and load the model using zstd compression to save space (with only pickle it was too big, I adpated the methods)

    def save(self, path="./Models/Model.pkl.zst"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_model_path = "./tempoarary_model.pkl"
        with open(temp_model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(temp_model_path, "rb") as f_in, open(path, "wb") as f_out:
            compressor = zstd.ZstdCompressor(level=19)
            f_out.write(compressor.compress(f_in.read()))
            print("Model saved in a compressed format")
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

    def load(self):
        path="./Model.pkl.zst"
        if not os.path.exists(path):
            print(f"No model found at the path {path}.")
            self.model = None
            return
        temp_model_path = "./tempoarary_model.pkl"
        try:
            #I decrompress the file
            with open(path, "rb") as f_in, open(temp_model_path, "wb") as f_out:
                decompressor = zstd.ZstdDecompressor()
                f_out.write(decompressor.decompress(f_in.read()))

            #I load the decompressed model
            with open(temp_model_path, "rb") as f:
                self.model = pickle.load(f)

            print("Model loaded successfully.")

        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.model = None

        finally:
            #I remove the temporary file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
