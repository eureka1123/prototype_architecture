#inspired by https://github.com/gsurma/cartpole - differences: eval and target network
#inspired by prototype architecture - differences: net architecture, losses

import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import os
import csv
import cv2
from PIL import Image
from helper_func import list_of_distances, list_of_norms
from graphics import visualize_prototypes
from cartpole import DQN as cartpole_DQN
import json

# GAMMA = .95
ENV_NAME = "CartPole-v1"
# LEARNING_RATE = .001
# BATCH_SIZE = 40
# EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.01
# EXPLORATION_DECAY = 0.999
# PROTOTYPE_SIZE_INNER = 100
# PROTOTYPE_SIZE = 30
# # TARGET_REPLACE_ITER = 10
# NUM_PROTOTYPES = 20
# MAX_MEMORY = 100000

run_for = 2000

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor

# num = 1
# param_dir = "param_"+str(num)
# if param_dir not in os.listdir():
#     os.mkdir(param_dir)
# param_dir +="/"

# weights_path = param_dir+"model_weights"
# ae_weights_path = param_dir+"ae_model_weights" 
# tweights_path = param_dir+"target_model_weights" 
# tae_weights_path = param_dir+"target_ae_model_weights"
# metadata_path = param_dir+"metadata"

# cartpole_weights_path = "cartpole_weights"

#weights for loss
# cl = .01
# l = 30 #.05
# l1 = .1#.05
# l2 = .1#.05

# #for model update
# tau = .01

# #for prototype differences
# dif_weight = 10
# parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
#                 "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
#                 "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
#                 "MAX_MEMORY":MAX_MEMORY,"weights_path":weights_path, "ae_weights_path":ae_weights_path, "tweights_path":tweights_path,
#                 "tae_weights_path":tae_weights_path,"metadata_path":metadata_path,"cartpole_weights_path":cartpole_weights_path,
#                 "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}

env = gym.make(ENV_NAME)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

def run_cartpole_dqn(parameters,trial = 0, train = False, threshold_step = 250, visualize = False):
    global GAMMA
    GAMMA = parameters["GAMMA"]
    global LEARNING_RATE
    LEARNING_RATE = parameters["LEARNING_RATE"]
    global BATCH_SIZE
    BATCH_SIZE = parameters["BATCH_SIZE"]
    global EXPLORATION_MAX
    EXPLORATION_MAX = parameters["EXPLORATION_MAX"]
    global EXPLORATION_MIN
    EXPLORATION_MIN = parameters["EXPLORATION_MIN"]
    global EXPLORATION_DECAY
    EXPLORATION_DECAY = parameters["EXPLORATION_DECAY"]
    global PROTOTYPE_SIZE_INNER
    PROTOTYPE_SIZE_INNER = parameters["PROTOTYPE_SIZE_INNER"] 
    global PROTOTYPE_SIZE
    PROTOTYPE_SIZE = parameters["PROTOTYPE_SIZE"]
    global NUM_PROTOTYPES 
    NUM_PROTOTYPES = parameters["NUM_PROTOTYPES"]
    global MAX_MEMORY
    MAX_MEMORY = parameters["MAX_MEMORY"]

    global param_dir
    param_dir = parameters["param_dir"]
    global weights_path
    weights_path = param_dir+"model_weights"
    global ae_weights_path
    ae_weights_path = param_dir+"ae_model_weights" 
    global tweights_path
    tweights_path = param_dir+"target_model_weights" 
    global tae_weights_path
    tae_weights_path = param_dir+"target_ae_model_weights"
    global metadata_path
    metadata_path = param_dir+"metadata"

    global cartpole_weights_path
    cartpole_weights_path = parameters["cartpole_weights_path"]
    #weights for loss
    global cl
    cl = parameters["cl"]
    global l
    l = parameters["l"] #.05
    global l1
    l1 = parameters["l1"]#.05
    global l2
    l2 = parameters["l2"]#.05

    #for model update
    global tau
    tau = parameters["tau"]

    #for prototype differences
    global dif_weight
    dif_weight = parameters["dif_weight"]

    cartpole_dqn = cartpole_DQN(observation_size, action_size)
    cartpole_dqn.eval_net.load_state_dict(torch.load(cartpole_weights_path))
    for p in cartpole_dqn.eval_net.parameters():
        p.requires_grad = False

    dqn = DQN(observation_size, action_size,cartpole_dqn)

    criterion = loss_func
    run = 0
    step = 0
    display = False
    scores = []
    if not train and os.path.exists(weights_path) and os.path.exists(ae_weights_path):
        dqn.eval_net.load_state_dict(torch.load(weights_path))
        dqn.eval_net.autoencoder.load_state_dict(torch.load(ae_weights_path))

        state = env.reset()
        step = 0
        done = False
        while not done:
            step +=1
            if not visualize:
                env.render()
            action = return_action(dqn, state, train=False)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = -reward
            state = next_state
            if done:
                print("run: ", run, " score: ", step)
                env.close()
    
    else:
        while not display and run<run_for:
            if len(scores)>20 and sum(scores[-20:])/20 >= threshold_step:
                display = True
            done = False
            env = gym.make(ENV_NAME)
            run += 1
            state = env.reset()
            step = 0
            while not done:
                step +=1
                if False and display:
                    env.render()
                action = return_action(dqn, state)
                next_state, reward, done, info = env.step(action)
                if done:
                    reward = -reward
                learn(parameters,dqn, criterion, state, action, reward, next_state, done)

                state = next_state
                if done:
                    print("trial: ", trial, " run: ", run, " score: ", step)
                    scores.append(step)
                    env.close()
        if run >= run_for:
            print("unfinished")


    # if visualize:
    #     autoencoder = dqn.eval_net.autoencoder
    #     decoded_prototypes = []
    #     for i in range(len(dqn.eval_net.prototypes)):
    #         prototype = dqn.eval_net.prototypes[i]
    #         decoded_prototype = autoencoder.decode(prototype).data.numpy()
    #         decoded_prototypes.append(decoded_prototype)
    #         env.env.state = decoded_prototype
    #         action = return_action(dqn, decoded_prototype,train = False)
    #         img = env.render(mode='rgb_array')
    #         img = visualize_prototypes(decoded_prototype, action, img)
    #         if "prototypes" not in os.listdir(param_dir):
    #             os.mkdir(param_dir+"prototypes")
    #         cv2.imwrite(param_dir+'prototypes/prototype_{}.png'.format(i), img)
    #         env.close()

    #     np.savetxt(param_dir+"prototypes.csv",decoded_prototypes)

    return scores, dqn

class Autoencoder(nn.Module):
    def __init__(self, observation_size):
        super(Autoencoder, self).__init__()
        self.efc1 = nn.Linear(observation_size,PROTOTYPE_SIZE_INNER)
        self.efc2 = nn.Linear(PROTOTYPE_SIZE_INNER,PROTOTYPE_SIZE_INNER)
        self.efc3 = nn.Linear(PROTOTYPE_SIZE_INNER,PROTOTYPE_SIZE)

        self.dfc1 = nn.Linear(PROTOTYPE_SIZE,PROTOTYPE_SIZE_INNER)
        self.dfc2 = nn.Linear(PROTOTYPE_SIZE_INNER,PROTOTYPE_SIZE_INNER)
        self.dfc3 = nn.Linear(PROTOTYPE_SIZE_INNER,observation_size)

    def encode(self, inputs):
        x = F.relu(self.efc1(inputs))
        x = F.relu(self.efc2(x))
        x = F.relu(self.efc3(x))
        return x

    def decode(self, inputs):
        x = F.relu(self.dfc1(inputs))
        x = F.relu(self.dfc2(x))
        x = self.dfc3(x)
        return x 

    def forward(self, inputs):
        transform_input = self.encode(inputs)
        recon_input = self.decode(transform_input)
        return transform_input, recon_input

class Net(nn.Module):
    def __init__(self, observation_size, action_size, cartpole_dqn):
        super(Net, self).__init__()
        self.num_prototypes = NUM_PROTOTYPES
        self.autoencoder = Autoencoder(observation_size)
        self.prototypes = nn.Parameter(torch.stack([torch.rand(size = (PROTOTYPE_SIZE,), requires_grad = True) for i in range(self.num_prototypes)]))
        self.fc1 = nn.Linear(PROTOTYPE_SIZE, action_size)
        self.cartpole_dqn = cartpole_dqn

    def forward(self, inputs):
        transform_input, recon_input = self.autoencoder(inputs)
        prototypes_difs = list_of_distances(transform_input, self.prototypes)
        feature_difs = list_of_distances(self.prototypes, transform_input)
      
        decoded_prototypes = self.autoencoder.decode(self.prototypes)
        # q_values = self.cartpole_dqn.eval_net(decoded_prototypes)
        # new_shape = list(prototypes_difs.shape)
        # new_shape.append(2)
        # new_shape = tuple(new_shape)
        # # print(prototypes_difs[0])
        # prototype_difs_reshape = (-np.repeat(prototypes_difs.detach()*dif_weight, 2,axis=1).reshape(new_shape))
        # # print(prototype_difs_reshape[0])
        
        # p_difs_sum = torch.sum(prototype_difs_reshape,1)
        # p_difs_sum_reshape = np.repeat(p_difs_sum.detach(),NUM_PROTOTYPES,axis = 0).reshape(new_shape)
        # prototype_difs_reshape/=p_difs_sum_reshape
        
        # output=prototype_difs_reshape*q_values
        
        # output=torch.sum(output,1)
        
        
        best_proto = self.prototypes[torch.argmin(prototypes_difs,dim=1)]
        decoded_prototype = self.autoencoder.decode(best_proto)
        output = self.cartpole_dqn.eval_net(decoded_prototype)
        # print(output[0], q_values[torch.argmin(prototypes_difs,dim=1)][0])
        # output = self.cartpole_dqn.eval_net(recon_input)
        # print(output)
        # output = self.fc1(best_proto)
        # print(output.shape)
        #   take direct q values and then take weighted
        # output = self.fc1(prototypes_difs)
        
        return transform_input, recon_input, self.prototypes, output, prototypes_difs, feature_difs

class DQN(object):
    def __init__(self, observation_size, action_size, cartpole_dqn):
        #maybe soft update, .001 as tau, 
        self.eval_net, self.target_net = Net(observation_size, action_size, cartpole_dqn), Net(observation_size, action_size, cartpole_dqn)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.memory = []
        self.learn_step_counter = 0
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_size
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)

def loss_func(parameters,transform_input, recon_input, input_target, output, output_target, prototypes_difs, feature_difs):
    # cl = .01
    # l = 10 #.05
    # l1 = .05#.05
    # l2 = .05#.05
    
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(output,output_target)
    recon_loss = list_of_norms(recon_input-input_target)
    recon_loss = torch.mean(recon_loss)
    r1_loss = torch.mean(torch.min(feature_difs,dim=1)[0])
    r2_loss = torch.mean(torch.min(prototypes_difs,dim=1)[0])

    total_loss = cl*mse_loss + l*recon_loss + l1*r1_loss + l2*r2_loss
    parameters["mse_loss"] = mse_loss.item()
    parameters["recon_loss"] = recon_loss.item()
    parameters["r1_loss"] = r1_loss.item()
    parameters["r2_loss"] = r2_loss.item()
    parameters["total_loss"] = total_loss.item()
    return mse_loss, recon_loss, r1_loss, r2_loss, total_loss

def learn(parameters,dqn, criterion, state, action, reward, next_state, done):
    # if dqn.learn_step_counter % TARGET_REPLACE_ITER == 0:
    #     dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
    # dqn.learn_step_counter += 1
    update(dqn.target_net, dqn.eval_net)
    dqn.eval_net.train()
    dqn.memory.append((FloatTensor([state]), LongTensor([[action]]), FloatTensor([reward]), FloatTensor([next_state]), FloatTensor([0 if done else 1])))
    dqn.memory = dqn.memory[-MAX_MEMORY:]

    if len(dqn.memory) < BATCH_SIZE:
        return 
    batch = random.sample(dqn.memory, BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

    batch_state  = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))
    batch_done = Variable(torch.cat(batch_done))

    transform_input, recon_input, prototypes, output, prototypes_difs, feature_difs = dqn.eval_net(batch_state)
    current_q_values = output.gather(1, batch_action).view(BATCH_SIZE)
    _,_,_,target_output,_,_ = dqn.target_net(batch_next_state)
    max_next_q_values = target_output.detach().max(1)[0]
    expected_q_values = ((GAMMA * max_next_q_values)*batch_done + batch_reward)

    mse_loss, recon_loss, r1_loss, r2_loss, loss = criterion(parameters,transform_input, recon_input, batch_state, current_q_values, expected_q_values, prototypes_difs, feature_difs)
    # print(mse_loss.item(), recon_loss.item(), r1_loss.item(), r2_loss.item(), loss.item())
    dqn.optimizer.zero_grad()
    loss.backward()
    
    dqn.optimizer.step()
    dqn.exploration_rate *= EXPLORATION_DECAY
    dqn.exploration_rate = max(EXPLORATION_MIN, dqn.exploration_rate)

def update(m1, m2):
    for key in list(m1.state_dict().keys()):
        key_split = key.split(".")
        m1_data = m1
        m2_data = m2
        for k in key_split:
            m1_data = m1_data.__getattr__(k)
            m2_data = m2_data.__getattr__(k)

        m1_data.data = (1-tau)*(m1_data.data) + tau*(m2_data.data)

def return_action(dqn, state, train = True):
    if train:
        if np.random.rand() < dqn.exploration_rate:
            return random.randrange(dqn.action_space)
    state_tensor = Variable(FloatTensor([state]))
    output = dqn.eval_net(state_tensor)
    if len(output)>1:
        q_values = output[3]
    else:
        q_values = output
    # prototypes_difs = output[4]
    # p_id = torch.argmin(prototypes_difs)
    return torch.argmax(q_values).item()

def plot_rewards(scores):
    runs = np.arange(len(scores))
    plt.plot(runs, scores)
    plt.xlabel("steps")
    plt.ylabel("scores")
    plt.savefig(param_dir+"scores.png")
    # plt.show()
    plt.close('all')

def generate_num_eps(parameters, num_eps = 20):
    len_learn = []
    best_scores = []
    for i in range(num_eps):
        print(i)
        scores,dqn = run_cartpole_dqn(parameters,i+1,True, 250, visualize=False)
        if len(best_scores) == 0 or sum(scores)/len(scores)>sum(best_scores)/len(best_scores):
            best_scores = scores
            best_dqn = dqn

        len_learn.append(len(scores))

    torch.save(best_dqn.eval_net.state_dict(), weights_path)
    torch.save(best_dqn.eval_net.autoencoder.state_dict(), ae_weights_path)
    torch.save(best_dqn.target_net.state_dict(), tweights_path)
    torch.save(best_dqn.target_net.autoencoder.state_dict(), tae_weights_path)

    plot_rewards(best_scores)
    # print("avg run length for original cartpole", sum(len_learn)/len(len_learn))
    eps = np.arange(num_eps)
    plt.plot(eps,len_learn)
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.savefig(param_dir+"eps_till_learn.png")
    visualize(best_dqn)
    parameters["avg_eps_till_learn"] = sum(len_learn)/len(len_learn)
    parameters["avg_ep_len"] = sum(best_scores)/len(best_scores)
    parameters["max_ep_len"] = max(best_scores)

    with open(metadata_path, 'w') as f:
        json.dump(parameters, f)
    # plt.show()

def visualize(dqn):
    autoencoder = dqn.eval_net.autoencoder
    decoded_prototypes = []
    for i in range(len(dqn.eval_net.prototypes)):
        prototype = dqn.eval_net.prototypes[i]
        decoded_prototype = autoencoder.decode(prototype).data.numpy()
        decoded_prototypes.append(decoded_prototype)
        env.env.state = decoded_prototype
        action = return_action(dqn, decoded_prototype,train = False)
        img = env.render(mode='rgb_array')
        img = visualize_prototypes(decoded_prototype, action, img)
        if "prototypes" not in os.listdir(param_dir):
            os.mkdir(param_dir+"prototypes")
        cv2.imwrite(param_dir+'prototypes/prototype_{}.png'.format(i), img)
        env.close()

    np.savetxt(param_dir+"prototypes.csv",decoded_prototypes)
# scores = run_cartpole_dqn(250)
# plot_rewards(scores)

# generate_num_runs(5)
if __name__ == "__main__":
    # scores = run_cartpole_dqn(True, 20, visualize=True)
    # plot_rewards(scores)
    generate_num_eps(parameters,5)

