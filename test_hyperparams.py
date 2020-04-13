from prototypes_cartpole import generate_num_eps
import os 

# GAMMA = .95
# LEARNING_RATE = .001
# BATCH_SIZE = 40
# EXPLORATION_MAX = 1.0
# EXPLORATION_MIN = 0.01
# EXPLORATION_DECAY = 0.999
# PROTOTYPE_SIZE_INNER = 100
# PROTOTYPE_SIZE = 30
# NUM_PROTOTYPES = 20
# MAX_MEMORY = 100000

# cartpole_weights_path = "cartpole_weights"

# #weights for loss
# cl = .005 
# l = 20 #.05
# l1 = .1#.05
# l2 = .1#.05

# #for model update
# tau = .01

#for prototype differences
dif_weight = 10

increase = 0
# for num in range(0,5):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	cl += increase*.1
# 	increase+=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# cl = .01
# increase = 0
# for num in range(5,10):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l += increase*5
# 	increase+=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# l = 20
# increase = 0
# for num in range(10,15):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l1 += increase*.1
# 	increase+=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# l1 = .1
# increase = 0
# for num in range(15,20):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l2 += increase*.1
# 	increase+=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# for num in range(21,22):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir ="temp/"

# 	l1+=.4
# 	l2 += .2
# 	l+=25

# 	# increase/=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# for num in range(22,23):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l1+=.4
# 	l2 += .2
# 	l+=25
# 	LEARNING_RATE = .0005
# 	# increase/=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# for num in range(23,24):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l1+=.4
# 	l2 += .2
# 	l+=25
# 	LEARNING_RATE = .0001
# 	# increase/=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# LEARNING_RATE = .001
# for num in range(24,25):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l1+=.4
# 	l2 += .2
# 	l=25
# 	PROTOTYPE_SIZE_INNER = 200
# 	# increase/=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)

# PROTOTYPE_SIZE_INNER = 100
# for num in range(25,30):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l1=2.1
# 	l2 += 1.1
# 	l=100
# 	BATCH_SIZE += 20
# 	# increase/=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)


# PROTOTYPE_SIZE_INNER = 100
# for num in range(25,30):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"

# 	l1+=.4
# 	l2 += .2
# 	l+=25
# 	BATCH_SIZE = 100
# 	# increase/=1

# 	parameters = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight}
	
# 	generate_num_eps(parameters,3)
# increase = 0
# for num in range(30,32):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
# 	cl +=.01*increase
# 	increase+=1
# 	l1=2.1
# 	l2 = 1.2
# 	l=145
# 	BATCH_SIZE = 60
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

# cl = .01
# l1=2.1
# l2 = 1.2
# l=145
# BATCH_SIZE = 60

# increase = 0
# LEARNING_RATE = .0001
# for num in range(32,34):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
	
# 	LEARNING_RATE += .0005*increase
# 	increase+=1
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

# increase = 0
# LEARNING_RATE = .001
# tau = .0001
# for num in range(34,38):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
# 	tau+=.0005*increase
# 	increase+=1
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

# increase = 0
# tau = .001
# for num in range(38,40):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
# 	GAMMA+=.02
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

# increase = 0
# for num in range(40,41):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
# 	MAX_MEMORY = 1000000
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

# increase = 0
# for num in range(41,45):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
# 	PROTOTYPE_SIZE_INNER+=100
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

# increase = 0
# PROTOTYPE_SIZE_INNER = 100
# for num in range(41,45):
# 	print("--------param: ", num,"---------")
# 	param_dir = "param_"+str(num)
# 	if param_dir not in os.listdir():
# 	    os.mkdir(param_dir)
# 	param_dir +="/"
# 	PROTOTYPE_SIZE += 10
# 	# increase/=1

# 	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
# 	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
# 	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
# 	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
# 	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
# 	params["weights_path"] = param_dir+"model_weights"
# 	params["ae_weights_path"] = param_dir+"ae_model_weights" 
# 	params["tweights_path"] = param_dir+"target_model_weights" 
# 	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
# 	params["metadata_path"] = param_dir+"metadata"
# 	# params = Params(parameters)
# 	generate_num_eps(params,2)

GAMMA = .97
LEARNING_RATE = .0001
BATCH_SIZE = 40
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.001
EXPLORATION_DECAY = 0.999
PROTOTYPE_SIZE_INNER = 100
PROTOTYPE_SIZE = 30
NUM_PROTOTYPES = 20
MAX_MEMORY = 100000

cartpole_weights_path = "cartpole_weights"

#weights for loss
cl = .016
l = 20 #.05
l1 = .1#.05
l2 = .1#.05

#for model update
tau = .0001

increase = 0
for num in range(22,24):
	print("--------param: ", num,"---------")
	param_dir = "param_"+str(num)
	if param_dir not in os.listdir():
	    os.mkdir(param_dir)
	param_dir +="/"
	tau+=.0004*increase
	# cl =.001*increase
	increase+=1

	params = {"GAMMA":GAMMA, "LEARNING_RATE": LEARNING_RATE, "BATCH_SIZE": BATCH_SIZE, 
	                "EXPLORATION_MAX": EXPLORATION_MAX, "EXPLORATION_MIN": EXPLORATION_MIN, "EXPLORATION_DECAY":EXPLORATION_DECAY,
	                "PROTOTYPE_SIZE_INNER": PROTOTYPE_SIZE_INNER, "PROTOTYPE_SIZE":PROTOTYPE_SIZE, "NUM_PROTOTYPES":NUM_PROTOTYPES,
	                "MAX_MEMORY":MAX_MEMORY,"param_dir":param_dir,"cartpole_weights_path":cartpole_weights_path,
	                "cl":cl, "l":l, "l1":l1, "l2":l2, "tau":tau, "dif_weight": dif_weight,}
	params["weights_path"] = param_dir+"model_weights"
	params["ae_weights_path"] = param_dir+"ae_model_weights" 
	params["tweights_path"] = param_dir+"target_model_weights" 
	params["tae_weights_path"] = param_dir+"target_ae_model_weights"
	params["metadata_path"] = param_dir+"metadata"
	# params = Params(parameters)
	generate_num_eps(params,20)

#exploration -> constant 
#fixed number of prototypes per action - dangerous assumption?
#improve dqn -> make sure it reaches 500
#weight matrix learnable 


