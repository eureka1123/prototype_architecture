import gym
from graphics import visualize_prototypes
# from prototypes_unavg_cartpole import DQN, return_action
from prototypes_cartpole import Net, return_action

import csv
import torch
import cv2
import numpy as np
from cartpole import DQN as cartpole_DQN
import os 
import json

param_dir = 'param_23'
with open(param_dir+"/metadata") as f:
    params = json.load(f)
# params = Params(params)
ENV_NAME = "CartPole-v1"
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

weights_path = params["weights_path"]#model_unavg_weights_50"
ae_weights_path = params["ae_weights_path"]#ae_model_unavg_weights_50"
cartpole_weights_path = params["cartpole_weights_path"]

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

p_ids = {}
interval = 10
same = 0
for e in range(10):
    states = []
    state = env.reset()
    step = 0
    done = False
    same = 0
    while not done:
        step +=1
        action = return_action(dqn, state, train=False)
        orig_action = return_action(cartpole_dqn, state, train=False)
        if action == orig_action:
            same += 1
        states.append((list(state), action, orig_action))
        next_state, reward, done, info = env.step(action)
        if done:
            reward = -reward
        state = next_state
        if done:
            print("run: ", e, " score: ", step)
            env.close()
    print(same, len(states), same/len(states))
    for i in range(len(states)):
        # print("pair",i)
        state, action, orig_action = states[i]
        if i%interval==0:
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
        if i%interval==0:
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
            if "pairs_using_dqn_20_eps_"+str(e) not in os.listdir(param_dir):
                os.mkdir(param_dir+"/pairs_using_dqn_20_eps_"+str(e))
            cv2.imwrite(param_dir+"/pairs_using_dqn_20_eps_{}/pair_{}.png".format(e,i), img)
            env.close()
    print(p_ids)



