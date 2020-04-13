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

ENV_NAME = "CartPole-v1"

run_for = 3000

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor

env = gym.make(ENV_NAME)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n


def test_net(params, trial, input_dqn, eps_count = 50):
    cartpole_dqn = cartpole_DQN(observation_size, action_size)
    cartpole_dqn.eval_net.load_state_dict(torch.load(params["cartpole_weights_path"]))

    dqn = Net(params, observation_size, action_size, cartpole_dqn)
    dqn.eval_net.load_state_dict(input_dqn.eval_net.state_dict())
    dqn.eval_net.autoencoder.load_state_dict(input_dqn.eval_net.autoencoder.state_dict())
  
    env = gym.make(ENV_NAME)
    scores = []
    for i in range(eps_count):
        done = False
        step = 0
        state = env.reset()
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
                print("test!!", "trial", trial, "eps: ", i, " score: ", step)
                scores.append(step)
                env.close()

    print("avg eps len", sum(scores)/len(scores))
    return scores

def train_net(params, trial = 0, train = False, threshold_step = 250, input_dqn = None, visualize = False):
    cartpole_dqn = cartpole_DQN(observation_size, action_size)
    cartpole_dqn.eval_net.load_state_dict(torch.load(params["cartpole_weights_path"]))

    for p in cartpole_dqn.eval_net.parameters():
        p.requires_grad = False

    dqn = Net(params,observation_size, action_size,cartpole_dqn)

    criterion = loss_func
    run = 0
    step = 0
    display = False
    if not train:
        dqn.eval_net.load_state_dict(input_dqn.eval_net.state_dict())
        dqn.eval_net.autoencoder.load_state_dict(input_dqn.eval_net.autoencoder.state_dict())

        env = gym.make(ENV_NAME)

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
                scores.append(step)
                env.close()
    
    else:
        while not display and run<run_for:
            run += 1
            done = False
            if run%100 == 0:
                scores = test_net(params, trial, dqn)
                if sum(scores)/len(scores) >= threshold_step:
                    display = True
                    done = True
            env = gym.make(ENV_NAME)
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
                learn(params,dqn, criterion, state, action, reward, next_state, done)

                state = next_state
                if done:
                    print("trial: ", trial, " run: ", run, " score: ", step)
                    env.close()
        if run >= run_for:
            print("unfinished")

    return run, dqn

class Autoencoder(nn.Module):
    def __init__(self, params,observation_size):
        super(Autoencoder, self).__init__()
        self.efc1 = nn.Linear(observation_size,params["PROTOTYPE_SIZE_INNER"])
        self.efc2 = nn.Linear(params["PROTOTYPE_SIZE_INNER"],params["PROTOTYPE_SIZE_INNER"])
        self.efc3 = nn.Linear(params["PROTOTYPE_SIZE_INNER"],params["PROTOTYPE_SIZE"])

        self.dfc1 = nn.Linear(params["PROTOTYPE_SIZE"],params["PROTOTYPE_SIZE_INNER"])
        self.dfc2 = nn.Linear(params["PROTOTYPE_SIZE_INNER"],params["PROTOTYPE_SIZE_INNER"])
        self.dfc3 = nn.Linear(params["PROTOTYPE_SIZE_INNER"],observation_size)

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

class ProtoNet(nn.Module):
    def __init__(self, params, observation_size, action_size, cartpole_dqn):
        super(ProtoNet, self).__init__()
        self.num_prototypes = params["NUM_PROTOTYPES"]
        self.autoencoder = Autoencoder(params,observation_size)
        self.prototypes = nn.Parameter(torch.stack([torch.rand(size = (params["PROTOTYPE_SIZE"],), requires_grad = True) for i in range(self.num_prototypes)]))
        # self.fc1 = nn.Linear(params["PROTOTYPE_SIZE"], action_size)
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
        # prototype_difs_reshape = (-np.repeat(prototypes_difs.detach()*dif_weight, 2,axis=1).reshape(new_shape))
        
        # p_difs_sum = torch.sum(prototype_difs_reshape,1)
        # p_difs_sum_reshape = np.repeat(p_difs_sum.detach(),NUM_PROTOTYPES,axis = 0).reshape(new_shape)
        # prototype_difs_reshape/=p_difs_sum_reshape
        
        # output=prototype_difs_reshape*q_values
        
        # output=torch.sum(output,1)
        
        
        best_proto = self.prototypes[torch.argmin(prototypes_difs,dim=1)]
        decoded_prototype = self.autoencoder.decode(best_proto)
        output = self.cartpole_dqn.eval_net(decoded_prototype)
        # output = self.cartpole_dqn.eval_net(recon_input)
        # output = self.fc1(best_proto)
        #   take direct q values and then take weighted
        # output = self.fc1(prototypes_difs)
        
        return transform_input, recon_input, self.prototypes, output, prototypes_difs, feature_difs

class Net(object):
    def __init__(self, params, observation_size, action_size, cartpole_dqn):
        self.eval_net, self.target_net = ProtoNet(params, observation_size, action_size, cartpole_dqn), ProtoNet(params,observation_size, action_size, cartpole_dqn)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.memory = []
        self.learn_step_counter = 0
        self.exploration_rate = params["EXPLORATION_MAX"]
        self.action_space = action_size
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=params["LEARNING_RATE"])

def loss_func(params,transform_input, recon_input, input_target, output, output_target, prototypes_difs, feature_difs):
    cl = params["cl"]
    l = params["l"]
    l1 = params["l1"]
    l2 = params["l2"]
    
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(output,output_target)
    recon_loss = list_of_norms(recon_input-input_target)
    recon_loss = torch.mean(recon_loss)
    r1_loss = torch.mean(torch.min(feature_difs,dim=1)[0])
    r2_loss = torch.mean(torch.min(prototypes_difs,dim=1)[0])

    total_loss = cl*mse_loss + l*recon_loss + l1*r1_loss + l2*r2_loss
    return mse_loss, recon_loss, r1_loss, r2_loss, total_loss

def learn(params,dqn, criterion, state, action, reward, next_state, done):
    update(params,dqn.target_net, dqn.eval_net)
    dqn.eval_net.train()
    dqn.memory.append((FloatTensor([state]), LongTensor([[action]]), FloatTensor([reward]), FloatTensor([next_state]), FloatTensor([0 if done else 1])))
    dqn.memory = dqn.memory[-params["MAX_MEMORY"]:]

    if len(dqn.memory) < params["BATCH_SIZE"]:
        return 
    batch = random.sample(dqn.memory, params["BATCH_SIZE"])
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)

    batch_state  = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))
    batch_done = Variable(torch.cat(batch_done))

    transform_input, recon_input, prototypes, output, prototypes_difs, feature_difs = dqn.eval_net(batch_state)
    current_q_values = output.gather(1, batch_action).view(params["BATCH_SIZE"])
    _,_,_,target_output,_,_ = dqn.target_net(batch_next_state)
    max_next_q_values = target_output.detach().max(1)[0]
    expected_q_values = ((params["GAMMA"] * max_next_q_values)*batch_done + batch_reward)

    mse_loss, recon_loss, r1_loss, r2_loss, loss = criterion(params,transform_input, recon_input, batch_state, current_q_values, expected_q_values, prototypes_difs, feature_difs)
    # print(mse_loss.item(), recon_loss.item(), r1_loss.item(), r2_loss.item(), loss.item())
    dqn.optimizer.zero_grad()
    loss.backward()
    
    dqn.optimizer.step()
    dqn.exploration_rate *= params["EXPLORATION_DECAY"]
    dqn.exploration_rate = max(params["EXPLORATION_MIN"], dqn.exploration_rate)

def update(params,m1, m2):
    for key in list(m1.state_dict().keys()):
        key_split = key.split(".")
        m1_data = m1
        m2_data = m2
        for k in key_split:
            m1_data = m1_data.__getattr__(k)
            m2_data = m2_data.__getattr__(k)

        m1_data.data = (1-params["tau"])*(m1_data.data) + params["tau"]*(m2_data.data)

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
    return torch.argmax(q_values).item()

def plot_rewards(params,scores):
    param_dir = params["param_dir"]
    runs = np.arange(len(scores))
    plt.plot(runs, scores)
    plt.xlabel("steps")
    plt.ylabel("scores")
    plt.savefig(param_dir+"scores.png")
    plt.clf()
    plt.close()

def generate_num_eps(params, num_eps = 20):
    len_learn = []
    best_scores = []
    for i in range(num_eps):
        print(i, "train")
        run,dqn = train_net(params,i+1,True, 400, visualize=False)
        print(i, "test")
        scores = test_net(params, i+1, dqn, eps_count = 50)

        if len(best_scores) == 0 or len(scores)< len(best_scores) or sum(scores)/len(scores)>sum(best_scores)/len(best_scores):
            best_scores = scores
            best_dqn = dqn

        len_learn.append(run)

    torch.save(best_dqn.eval_net.state_dict(), params["weights_path"])
    torch.save(best_dqn.eval_net.autoencoder.state_dict(), params["ae_weights_path"])
    torch.save(best_dqn.target_net.state_dict(), params["tweights_path"])
    torch.save(best_dqn.target_net.autoencoder.state_dict(), params["tae_weights_path"])

    plot_rewards(params,best_scores)
    eps = np.arange(num_eps)
    plt.plot(eps,len_learn)
    plt.xlabel("episodes")
    plt.ylabel("steps")
    plt.savefig(params["param_dir"]+"eps_till_learn.png")
    plt.clf()
    plt.close()
    visualize(params,best_dqn)
    params["avg_eps_till_learn"] = sum(len_learn)/len(len_learn)
    params["avg_ep_len"] = sum(best_scores)/len(best_scores)
    params["max_ep_len"] = max(best_scores)

    with open(params["metadata_path"], 'w') as f:
        json.dump(params, f)

def visualize(params,dqn):
    param_dir = params["param_dir"]
    autoencoder = dqn.eval_net.autoencoder
    decoded_prototypes = []
    for i in range(len(dqn.eval_net.prototypes)):
        prototype = dqn.eval_net.prototypes[i]
        decoded_prototype = autoencoder.decode(prototype).data.numpy()
        decoded_prototypes.append(decoded_prototype)
        env.env.state = decoded_prototype
        action = return_action(dqn, decoded_prototype,train = False)
        try:
            img = env.render(mode='rgb_array')
            img = visualize_prototypes(decoded_prototype, action, img)
            if "prototypes" not in os.listdir(param_dir):
                os.mkdir(param_dir+"prototypes")
            cv2.imwrite(param_dir+'prototypes/prototype_{}.png'.format(i), img)
            env.close()
        except:
            env.close()

    np.savetxt(param_dir+"prototypes.csv",decoded_prototypes)

if __name__ == "__main__":
    generate_num_eps(parameters,5)

