import gym
from graphics import visualize_prototypes
from prototypes_cartpole import Net, return_action
import csv
import torch
import cv2
import numpy as np
from cartpole import DQN as cartpole_DQN
import os 
import json

ENV_NAME = "CartPole-v1"
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

param_dir = 'param_23'
with open(param_dir+"/metadata") as f:
    params = json.load(f)

weights_path = param_dir+"/model_weights"
ae_weights_path = param_dir+"/ae_model_weights"
cartpole_weights_path = "cartpole_weights"

states_path = "states.csv"

env = gym.make(ENV_NAME)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

cartpole_dqn = cartpole_DQN(observation_size, action_size)
cartpole_dqn.eval_net.load_state_dict(torch.load(cartpole_weights_path))

dqn = Net(params,observation_size, action_size, cartpole_dqn)
dqn.eval_net.load_state_dict(torch.load(weights_path))
dqn.eval_net.autoencoder.load_state_dict(torch.load(ae_weights_path))
autoencoder = dqn.eval_net.autoencoder

for p in cartpole_dqn.eval_net.parameters():
    p.requires_grad = False

states = []
state = env.reset()
step = 0
done = False

while not done:
    step +=1
    action = return_action(dqn, state, train=False)
    orig_action = return_action(cartpole_dqn, state, train=False)
    states.append(list(state))
    next_state, reward, done, info = env.step(action)
    if done:
        reward = -reward
    state = next_state
    if done:
        env.close()

steps_num = len(states)
print(len(states))

old_states = []
with open (states_path, "r") as f:
    reader = csv.reader(f)
    old_states = list(list(i) for i in reader)
old_states = states.extend([[float(i) for i in j] for j in old_states])

p_ids = {}
print(len(states))

def categorize(i):
    print("state",i)
    state = states[i]
    action = return_action(dqn, state, train=False)
    orig_action = return_action(cartpole_dqn, state, train=False)

    
    env.env.state = state
    state_img = env.render(mode='rgb_array')
    state_img = visualize_prototypes(state, orig_action, state_img)
    state_img = cv2.copyMakeBorder(
        state_img,
        top=0,
        bottom=0,
        left=0,
        right=2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0,0,0)
    )

    env.close()

    state = FloatTensor([state])
    transform_input, recon_input, prototypes, output, prototypes_difs, feature_difs = dqn.eval_net(state)
    p_id = torch.argmin(prototypes_difs)
    p_id = int(p_id.data.numpy())
    p_ids.setdefault(p_id,0)
    p_ids[p_id]+=1
    prototype = dqn.eval_net.prototypes[p_id]
    
    decoded_prototype = autoencoder.decode(prototype).data.numpy()
    env.env.state = decoded_prototype
    proto_img = env.render(mode='rgb_array')
    proto_img = visualize_prototypes(decoded_prototype, action, proto_img)

    org = (400, 50) 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = .4
    thickness = 1
    color = (0,255,0)
    proto_img = cv2.putText(proto_img, "prototype "+str(p_id), org, font, fontScale, color, thickness, cv2.LINE_AA)

    img = np.concatenate((state_img, proto_img), axis=1)
    parent_dir = param_dir+"/prototypes_categorized_"+str(params["NUM_PROTOTYPES"])
    if "prototypes_categorized_"+str(params["NUM_PROTOTYPES"]) not in os.listdir(param_dir):
        os.mkdir(parent_dir)
    if "prototype_{}".format(p_id) not in os.listdir(parent_dir):
        os.mkdir(parent_dir+"/prototype_{}".format(p_id))
    cv2.imwrite(parent_dir+'/prototype_{}/state_{}.png'.format(p_id,i), img)
    env.close()

for i in range(0,steps_num):#len(states),50):
    categorize(i)

for i in range(steps_num, len(states),50):
    categorize(i)



print(p_ids)