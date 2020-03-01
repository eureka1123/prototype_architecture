#inspired by https://github.com/gsurma/cartpole
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
import matplotlib.pyplot as plt

GAMMA = .95
ENV_NAME = "CartPole-v1"
LEARNING_RATE = .001
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.999
TARGET_REPLACE_ITER = 10

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor

check_interval = 50
def run_cartpole_dqn(threshold_step = 250):
    weights_path = "cartpole_weights"
    states_path = "states.csv"
    env = gym.make(ENV_NAME)
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn = DQN(observation_size, action_size)

    criterion = nn.MSELoss()
    run = 0
    step = 0
    display = False
    scores = []
    states = []
    if True and os.path.exists(weights_path):
        dqn.eval_net.load_state_dict(torch.load(weights_path))
        state = env.reset()
        step = 0
        done = False
        while not done:
            step +=1
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
        while not display:
            if sum(scores[-check_interval:])/check_interval >= threshold_step:
                display = True
            done = False
            env = gym.make(ENV_NAME)
            run += 1
            state = env.reset()
            # state = np.reshape(state, [1, observation_size])
            step = 0
            while not done:
                step +=1
                if display:
                    break
                    env.render()
                action = return_action(dqn, state)
                next_state, reward, done, info = env.step(action)
                states.append(list(next_state))
                # next_state = np.reshape(next_state, [1, observation_size])
                if done:
                    reward = -reward
                learn(dqn, criterion, state, action, reward, next_state, done)

                state = next_state
                if done:
                    scores.append(step)
                    print("run: ", run, " score: ", step)
                    env.close()

        torch.save(dqn.eval_net.state_dict(), weights_path)
        print("num states", len(states))
        with open(states_path, "w") as f:
            writer = csv.writer(f)
            writer.writerows(states)
    return scores
    

class Net(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(observation_size, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50, action_size) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(self, observation_size, action_size):
        self.eval_net, self.target_net = Net(observation_size, action_size), Net(observation_size, action_size)
        self.memory = []
        self.learn_step_counter = 0
        self.exploration_rate = EXPLORATION_MAX
        self.action_space = action_size
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)


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

    current_q_values = dqn.eval_net(batch_state).gather(1, batch_action).view(BATCH_SIZE)
    max_next_q_values = dqn.target_net(batch_next_state).detach().max(1)[0]
    expected_q_values = ((GAMMA * max_next_q_values)*batch_done + batch_reward)
    loss = criterion(current_q_values, expected_q_values)
    dqn.optimizer.zero_grad()
    loss.backward()
    
    dqn.optimizer.step()
    dqn.exploration_rate *= EXPLORATION_DECAY
    dqn.exploration_rate = max(EXPLORATION_MIN, dqn.exploration_rate)


def return_action(dqn, state, train=True):
    if train and np.random.rand() < dqn.exploration_rate:
        return random.randrange(dqn.action_space)
    state_tensor = Variable(FloatTensor([state]))
    q_values = dqn.eval_net(state_tensor)
    return torch.argmax(q_values).item()

def plot_rewards(scores):
    runs = np.arange(len(scores))
    plt.plot(scores)
    plt.xlabel("runs")
    plt.ylabel("scores")
    plt.savefig("cartpole_orig_scores_temp.png")
    plt.show()

def generate_num_runs(num_runs = 20):
    len_learn = []
    for i in range(num_runs):
        print(i)
        scores = run_cartpole_dqn(250)
        len_learn.append(len(scores))
    plot_rewards(scores)
    print("avg run length for original cartpole", sum(len_learn)/len(len_learn))
    plt.plot(len_learn)
    plt.xlabel("episodes")
    plt.ylabel("runs")
    plt.savefig("cartpole_orig_runs_till_learn.png")
    plt.show()
if __name__ == "__main__":
    scores = run_cartpole_dqn(300)
# plot_rewards(scores)

# generate_num_runs(2)
