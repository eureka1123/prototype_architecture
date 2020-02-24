import gym
from graphics import visualize_prototypes
from prototypes_cartpole import DQN, return_action
import csv
import torch
import cv2
import numpy as np

ENV_NAME = "CartPole-v1"
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

weights_path = "model_weights"
ae_weights_path = "ae_model_weights"
env = gym.make(ENV_NAME)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn = DQN(observation_size, action_size)
dqn.eval_net.load_state_dict(torch.load(weights_path))
dqn.eval_net.autoencoder.load_state_dict(torch.load(ae_weights_path))
autoencoder = dqn.eval_net.autoencoder
p_ids = {}
interval = 50

for e in range(1):
    states = []
    state = env.reset()
    step = 0
    done = False
    while not done:
        step +=1
        action = return_action(dqn, state, train=False)
        states.append(list(state))
        next_state, reward, done, info = env.step(action)
        if done:
            reward = -reward
        state = next_state
        if done:
            print("run: ", e, " score: ", step)
            env.close()

    for i in range(len(states)):
        print("pair",i)
        state = states[i]
        if i%interval==0:
            env.env.state = state
            state_img = env.render(mode='rgb_array')
            state_img = visualize_prototypes(state, state_img)
            env.close()

        state = FloatTensor([state])
        transform_input, recon_input, prototypes, output, prototypes_difs, feature_difs = dqn.eval_net(state)
        print(prototypes_difs)
        p_id = torch.argmin(prototypes_difs)
        print(p_id)
        p_id = int(p_id.data.numpy())
        p_ids.setdefault(p_id,0)
        p_ids[p_id]+=1
        prototype = dqn.eval_net.prototypes[p_id]
        if i%interval==0:
            decoded_prototype = autoencoder.decode(prototype).data.numpy()
            env.env.state = decoded_prototype
            proto_img = env.render(mode='rgb_array')
            proto_img = visualize_prototypes(decoded_prototype, proto_img)

            org = (400, 50) 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = .4
            thickness = 1
            color = (0,255,0)
            proto_img = cv2.putText(proto_img, "prototype "+str(p_id), org, font, fontScale, color, thickness, cv2.LINE_AA)

            img = np.concatenate((state_img, proto_img), axis=1)
            cv2.imwrite('pairs/episode_{}_pair_{}.png'.format(e,i), img)
            env.close()
    print(p_ids)



