import gym
from graphics import visualize_prototypes
from prototypes_unavg_cartpole import DQN, return_action
import csv
import torch
import cv2
import numpy as np
from cartpole import DQN as cartpole_DQN
import os 

ENV_NAME = "CartPole-v1"
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

weights_path = "model_unavg_weights"
ae_weights_path = "ae_model_unavg_weights"
cartpole_weights_path = "cartpole_weights"

states_path = "states.csv"

env = gym.make(ENV_NAME)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

cartpole_dqn = cartpole_DQN(observation_size, action_size)
cartpole_dqn.eval_net.load_state_dict(torch.load(cartpole_weights_path))

dqn = DQN(observation_size, action_size, cartpole_dqn)
dqn.eval_net.load_state_dict(torch.load(weights_path))
dqn.eval_net.autoencoder.load_state_dict(torch.load(ae_weights_path))
autoencoder = dqn.eval_net.autoencoder

for p in cartpole_dqn.eval_net.parameters():
    p.requires_grad = False

with open (states_path, "r") as f:
    reader = csv.reader(f)
    states = list(list(i) for i in reader)
states = [[float(i) for i in j] for j in states]

p_ids = {}
print(len(states))
for i in range(0,len(states),100):
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
    if "prototype_{}".format(p_id) not in os.listdir('prototypes_categorized'):
        os.mkdir("prototypes_categorized/prototype_{}".format(p_id))
    cv2.imwrite('prototypes_categorized/prototype_{}/state_{}.png'.format(p_id,i), img)
    env.close()

print(p_ids)