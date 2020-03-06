from prototypes_cartpole import generate_num_eps
import os 

GAMMA = .95
LEARNING_RATE = .001
BATCH_SIZE = 40
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999
PROTOTYPE_SIZE_INNER = 100
PROTOTYPE_SIZE = 30
NUM_PROTOTYPES = 20
MAX_MEMORY = 100000

cartpole_weights_path = "cartpole_weights"

#weights for loss
cl = .01 
l = 20 #.05
l1 = .1#.05
l2 = .1#.05

#for model update
tau = .01

#for prototype differences
dif_weight = 10

increase = 0
for num in range(0,5):
	print("--------param: ", num,"---------")
	param_dir = "param_"+str(num)
	if param_dir not in os.listdir():
	    os.mkdir(param_dir)
	param_dir +="/"

	cl += increase*.1
	increase+=1

	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
	generate_num_eps(parameters,3)

cl = .01
increase = 0
for num in range(5,10):
	print("--------param: ", num,"---------")
	param_dir = "param_"+str(num)
	if param_dir not in os.listdir():
	    os.mkdir(param_dir)
	param_dir +="/"

	l += increase*5
	increase+=1

	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
	generate_num_eps(parameters,3)

l = 20
increase = 0
for num in range(10,15):
	print("--------param: ", num,"---------")
	param_dir = "param_"+str(num)
	if param_dir not in os.listdir():
	    os.mkdir(param_dir)
	param_dir +="/"

	l1 += increase*.1
	increase+=1

	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
	generate_num_eps(parameters,3)

l1 = .1
increase = 0
for num in range(15,20):
	print("--------param: ", num,"---------")
	param_dir = "param_"+str(num)
	if param_dir not in os.listdir():
	    os.mkdir(param_dir)
	param_dir +="/"

	l2 += increase*.1
	increase+=1

	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
	generate_num_eps(parameters,3)
