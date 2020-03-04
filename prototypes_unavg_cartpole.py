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
from PIL import Image
import cv2
from helper_func import list_of_distances, list_of_norms
from graphics import visualize_prototypes
from cartpole import DQN as cartpole_DQN

GAMMA = .95
ENV_NAME = "CartPole-v1"
LEARNING_RATE = .001
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.999
PROTOTYPE_SIZE = 10
TARGET_REPLACE_ITER = 40
NUM_PROTOTYPES = 10

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor

weights_path = "new_model_unavg_weights"
ae_weights_path = "new_ae_model_unavg_weights"
cartpole_weights_path = "cartpole_weights"

def run_cartpole_dqn(train = False, threshold_step = 250, visualize = False):

    env = gym.make(ENV_NAME)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n

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
            orig_action = return_action(cartpole_dqn, state, train=False)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = -reward
            state = next_state
            if done:
                print("run: ", run, " score: ", step)
                env.close()
    
    else:
        while not display:
            if sum(scores[-8:])/8 >= threshold_step:
                display = True
            done = False
            env = gym.make(ENV_NAME)
            run += 1
            state = env.reset()
            step = 0
            while not done:
                step +=1
                if display and not visualize:
                    env.render()
                action = return_action(dqn, state)
                next_state, reward, done, info = env.step(action)
                if done:
                    reward = -reward
                learn(dqn, criterion, state, action, reward, next_state, done)

                state = next_state
                if done:
                    print("run: ", run, " score: ", step)
                    scores.append(step)
                    env.close()

        torch.save(dqn.eval_net.state_dict(), weights_path)
        torch.save(dqn.eval_net.autoencoder.state_dict(), ae_weights_path)


    if visualize:
        autoencoder = dqn.eval_net.autoencoder
        decoded_prototypes = []
        for i in range(len(dqn.eval_net.prototypes)):
            prototype = dqn.eval_net.prototypes[i]
            decoded_prototype = autoencoder.decode(prototype).data.numpy()
            decoded_prototypes.append(decoded_prototype)
            env.env.state = decoded_prototype
            img = env.render(mode='rgb_array')
            img = visualize_prototypes(decoded_prototype, img)
            cv2.imwrite('prototypes_unavg/prototype_unavg_{}.png'.format(i), img)
            env.close()

        np.savetxt("prototypes_unavg.csv",decoded_prototypes)

    return scores

class Autoencoder(nn.Module):
    def __init__(self, observation_size):
        super(Autoencoder, self).__init__()
        self.efc1 = nn.Linear(observation_size,PROTOTYPE_SIZE)
        self.efc2 = nn.Linear(PROTOTYPE_SIZE,PROTOTYPE_SIZE)
        self.efc3 = nn.Linear(PROTOTYPE_SIZE,PROTOTYPE_SIZE)

        self.dfc1 = nn.Linear(PROTOTYPE_SIZE,PROTOTYPE_SIZE)
        self.dfc2 = nn.Linear(PROTOTYPE_SIZE,PROTOTYPE_SIZE)
        self.dfc3 = nn.Linear(PROTOTYPE_SIZE,observation_size)

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
        self.cartpole_dqn = cartpole_dqn

        self.fc1 = nn.Linear(PROTOTYPE_SIZE, action_size)

    def forward(self, inputs):
        transform_input, recon_input = self.autoencoder(inputs)
        prototypes_difs = list_of_distances(transform_input,self.prototypes)
        feature_difs = list_of_distances(self.prototypes,transform_input)
        best_proto = self.prototypes[torch.argmin(prototypes_difs,dim=1)]
        output = self.fc1(best_proto)
        
        return transform_input, recon_input, self.prototypes, output, prototypes_difs, feature_difs

class DQN(object):
    def __init__(self, observation_size, action_size, cartpole_dqn):
        #maybe soft update, .001 as tau, 
        self.eval_net, self.target_net = Net(observation_size, action_size, cartpole_dqn), Net(observation_size, action_size, cartpole_dqn)
        self.memory = []
        self.learn_step_counter = 0
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_size
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)

def list_of_distances(X,Y):
    XX = list_of_norms(X)
    XX = XX.view(-1,1)
    YY = list_of_norms(Y)
    YY = YY.view(1,-1)
    output = XX + YY - 2*torch.matmul(X,Y.transpose(0,1))
    return output

def list_of_norms(X):
    x = torch.pow(X,2)
    x = x.view(x.shape[0],-1)
    x = x.sum(1)
    return x

def loss_func(transform_input, recon_input, input_target, output, output_target, prototypes_difs, feature_difs):
    cl = 20
    l = 1 #.05
    l1 = 1#.05
    l2 = 1#.05
    
    mse_loss_fn = nn.MSELoss()
    mse_loss = mse_loss_fn(output,output_target)
    recon_loss = list_of_norms(recon_input-input_target)
    recon_loss = torch.mean(recon_loss)
    r1_loss = torch.mean(torch.min(feature_difs,dim=1)[0])
    r2_loss = torch.mean(torch.min(prototypes_difs,dim=1)[0])

    total_loss = cl*mse_loss + l*recon_loss + l1*r1_loss + l2*r2_loss
    return mse_loss, recon_loss, r1_loss, r2_loss, total_loss

def learn(dqn, criterion, state, action, reward, next_state, done):
    if dqn.learn_step_counter % TARGET_REPLACE_ITER == 0:
        dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
    dqn.learn_step_counter += 1

    dqn.eval_net.train()
    dqn.memory.append((FloatTensor([state]), LongTensor([[action]]), FloatTensor([reward]), FloatTensor([next_state]), FloatTensor([0 if done else 1])))

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
    _,_,_,target_output,_,_ = dqn.eval_net(batch_next_state)
    max_next_q_values = target_output.detach().max(1)[0]
    expected_q_values = ((GAMMA * max_next_q_values)*batch_done + batch_reward)

    mse_loss, recon_loss, r1_loss, r2_loss, loss = criterion(transform_input, recon_input, batch_state, current_q_values, expected_q_values, prototypes_difs, feature_difs)
    dqn.optimizer.zero_grad()
    loss.backward()
    
    dqn.optimizer.step()
    dqn.exploration_rate *= EXPLORATION_DECAY
    dqn.exploration_rate = max(EXPLORATION_MIN, dqn.exploration_rate)


def return_action(dqn, state, train = True):
    if train:
        if np.random.rand() < dqn.exploration_rate:
            return random.randrange(dqn.action_space)
    state_tensor = Variable(FloatTensor([state]))
    output = dqn.eval_net(state_tensor)
    if len(output)>1:
        q_values = dqn.eval_net(state_tensor)[3]
    else:
        q_values = dqn.eval_net(state_tensor)
    return torch.argmax(q_values).item()

def plot_rewards(scores):
    runs = np.arange(len(scores))
    plt.plot(runs, scores)
    plt.xlabel("runs")
    plt.ylabel("scores")
    plt.savefig("cartpole_proto_scores.png")
    plt.show()

def generate_num_runs(num_runs = 20):
    len_learn = []
    for i in range(num_runs):
        print(i)
        scores = run_cartpole_dqn(True, 250, visualize=False)
        len_learn.append(len(scores))
    plot_rewards(scores)
    print("avg run length for original cartpole", sum(len_learn)/len(len_learn))
    plt.plot(len_learn)
    plt.xlabel("episodes")
    plt.ylabel("runs")
    plt.savefig("cartpole_orig_runs_till_learn.png")
    plt.show()

# scores = run_cartpole_dqn(250)
# plot_rewards(scores)

# generate_num_runs(5)

if __name__ == "__main__":
    scores = run_cartpole_dqn(True, 250, visualize=True)
# plot_rewards(scores)


