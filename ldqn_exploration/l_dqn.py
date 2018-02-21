from __future__ import division
import time
import datetime
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import l1_l2_regularizer 
import matplotlib.pyplot as plt
import scipy.misc
import os

save_at_all = True

N_ACTIONS = 3
S_DIM = 2
HIDDEN_UNITS = [64, 128]
#HIDDEN_UNITS = [8, 16]
EGREEDY = "eps"
DORA = "dora"
COUNTER = "counter"
SOFTMAX = "softmax"
EGREEDY_DORA = "eps_dora"
agent = DORA

success_bonus = 5
ORIG = "orig"
MODIFIED = "modf"
BINARY = "binary"
REWARD_MODE = ORIG

load_model = False #Whether to load a saved model.
render_each = None # 50
print_rate = 10
render_test_rate = float("inf") #5
save_rate = 1000

env = gym.make('MountainCar-v0')

class Qnetwork():
	def __init__(self):
		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		
		# self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
		# self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
		# self.conv1 = slim.conv2d( \
		#     inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
		# self.conv2 = slim.conv2d( \
		#     inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
		# self.conv3 = slim.conv2d( \
		#     inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
		# self.conv4 = slim.conv2d( \
		#     inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
		

		self.stateInput = tf.placeholder(shape=[None,S_DIM], dtype=tf.float32)
		self.hidden = slim.fully_connected(inputs=self.stateInput, num_outputs = HIDDEN_UNITS[0], activation_fn=tf.tanh)
		self.hidden2 = slim.fully_connected(inputs=self.hidden, num_outputs = HIDDEN_UNITS[1], activation_fn=tf.tanh)
		self.out = slim.fully_connected(inputs=self.hidden2, num_outputs = N_ACTIONS, activation_fn=tf.identity,biases_initializer=None)

		#We take the output from the final convolutional layer and split it into separate advantage and value streams.
		#self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
		#self.streamA = slim.flatten(self.streamAC)
		#self.streamV = slim.flatten(self.streamVC)
		#xavier_init = tf.contrib.layers.xavier_initializer()
		#self.AW = tf.Variable(xavier_init([h_size//2,env.actions]))
		#self.VW = tf.Variable(xavier_init([h_size//2,1]))
		#self.Advantage = tf.matmul(self.streamA,self.AW)
		#self.Value = tf.matmul(self.streamV,self.VW)
		
		#Then combine them together to get our final Q-values.
		#self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
		#self.predict = tf.argmax(self.Qout,1)
		self.predict = tf.argmax(self.out,1)
		self.predict_val = tf.reduce_max(self.out,reduction_indices=[1])
		
		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,N_ACTIONS,dtype=tf.float32)
		
		self.Q = tf.reduce_sum(tf.mul(self.out, self.actions_onehot), reduction_indices=[1])
		
		self.td_error = tf.square(self.targetQ - self.Q)
		self.loss = tf.reduce_mean(self.td_error)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.updateModel = self.trainer.minimize(self.loss)


class Enetwork():
	def __init__(self):
		#The network recieves a frame from the game, flattened into an array.
		#It then resizes it and processes it through four convolutional layers.
		
		# self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
		# self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
		# self.conv1 = slim.conv2d( \
		#     inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
		# self.conv2 = slim.conv2d( \
		#     inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
		# self.conv3 = slim.conv2d( \
		#     inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
		# self.conv4 = slim.conv2d( \
		#     inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
		

		self.stateInput = tf.placeholder(shape=[None,S_DIM], dtype=tf.float32)
		self.hidden = slim.fully_connected(inputs=self.stateInput, num_outputs = HIDDEN_UNITS[0], activation_fn=tf.tanh)
		self.hidden2 = slim.fully_connected(inputs=self.hidden, num_outputs = HIDDEN_UNITS[1], activation_fn=tf.tanh)
		self.out = slim.fully_connected(inputs=self.hidden2, num_outputs = N_ACTIONS, 
			weights_initializer=init_ops.zeros_initializer, activation_fn=tf.nn.sigmoid	,biases_initializer=None)

		#We take the output from the final convolutional layer and split it into separate advantage and value streams.
		#self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
		#self.streamA = slim.flatten(self.streamAC)
		#self.streamV = slim.flatten(self.streamVC)
		#xavier_init = tf.contrib.layers.xavier_initializer()
		#self.AW = tf.Variable(xavier_init([h_size//2,env.actions]))
		#self.VW = tf.Variable(xavier_init([h_size//2,1]))
		#self.Advantage = tf.matmul(self.streamA,self.AW)
		#self.Value = tf.matmul(self.streamV,self.VW)
		
		#Then combine them together to get our final Q-values.
		#self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
		#self.predict = tf.argmax(self.Qout,1)
		self.predict = tf.argmax(self.out,1)
		self.predict_val = tf.reduce_max(self.out,reduction_indices=[1])

		#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
		self.targetE = tf.placeholder(shape=[None],dtype=tf.float32)
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,N_ACTIONS,dtype=tf.float32)
		
		self.E = tf.reduce_sum(tf.mul(self.out, self.actions_onehot), reduction_indices=[1])
		
		self.td_error = tf.square(self.targetE - self.E)
		self.loss = tf.reduce_mean(self.td_error)
		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.updateModel = self.trainer.minimize(self.loss)



class experience_buffer():
	def __init__(self, buffer_size = 50000):
		self.buffer = []
		self.buffer_size = buffer_size
	
	def add(self,experience):
		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
		self.buffer.extend(experience)
			
	def sample(self,size):
		return np.reshape(np.array(random.sample(self.buffer,size)),[size,6])

def processState(states):
	return states

def updateTargetGraph(tfVars, net):
	total_vars = len(tfVars)
	if net == 0:
		tfVars = tfVars[0:total_vars//2]
	else:
		tfVars = tfVars[total_vars//2:]

	net_vars = len(tfVars)

	op_holder = []
	for idx,var in enumerate(tfVars[0:net_vars//2]):
		op_holder.append(tfVars[idx+net_vars//2].assign(var.value()))
	return op_holder

def updateTarget(op_holder, sess):
	for op in op_holder:
		sess.run(op)

def get_bins(dim, bins_num):
	return np.linspace(env.observation_space.low[dim], env.observation_space.high[dim], bins_num + 1)

def discretize(state, shape):
	res = []
	for dim, bins_num in enumerate(shape):
		bins = get_bins(dim, bins_num)
		res.append(np.digitize(state[dim], bins) - 1)
	return tuple(res)

def is_success(s,d):
	return d
	return s[0]>0.45

def process_reward(r, s_next, d):
	if REWARD_MODE == ORIG:
		return r

	elif REWARD_MODE == MODIFIED:
		if is_success(s_next,d):
			return s_next[0]*success_bonus
		else:
			return s_next[0]

	elif REWARD_MODE == BINARY:
		return 1 if is_success(s_next,d) else 0

def action_selection(sess, state, temp, e, mainQN, mainEN=None):
	if agent in [DORA, COUNTER]:
		qs,es = sess.run([mainQN.out, mainEN.out], feed_dict={mainQN.stateInput:[state],mainEN.stateInput:[state]})
		#es = sess.run(mainEN.out, feed_dict={mainEN.stateInput:[s]})[0]
		a = np.argmax(qs - temp*np.log(-np.log(es)))
	
	elif agent == EGREEDY:
		if np.random.rand(1) < e: #or total_steps < pre_train_steps:
			a = np.random.randint(0,N_ACTIONS)
		else:
			a = sess.run(mainQN.predict,feed_dict={mainQN.stateInput:[state]})[0]
	
	elif agent == SOFTMAX:
		qs = sess.run(mainQN.out, feed_dict={mainQN.stateInput:[state]})[0]
		a = np.random.choice(3, p=np.exp(qs/temp)/np.sum(np.exp(qs/temp)))

	elif agent == EGREEDY_DORA:
		greedy_a = sess.run(mainQN.predict,feed_dict={mainQN.stateInput:[state]})[0]
		ps = [1/e if a!=greedy_a else 1-e + (e/N_ACTIONS) for a in range(N_ACTIONS)]
		es = sess.run(mainEN.out, feed_dict={mainEN.stateInput:[state]})
		a = np.argmax(ps/-np.log(es))

	return a

def run_test_episode(sess, mainQN, mainEN, env, n=1, render=False, random=False):
	rList = []
	sList = []
	for i in range(n):

		s = env.reset()
		s = processState(s)
		d = False
		j = 0

		sList.append(False)
		total_r = 0

		if i==0 and render:
			print("RENDERING TEST")

		while j < max_epLength: #If the agent takes longer than max_epLength moves to reach either of the blocks, end the trial.

			if i==0 and render:
				env.render()

			j+=1
			#Choose an action by greedily (with e chance of random action) from the Q-network
			
			if not random:
				vals = sess.run(mainQN.out,feed_dict={mainQN.stateInput:[s]})
				evals = sess.run(mainEN.out,feed_dict={mainEN.stateInput:[s]})
				#print(vals,-np.log(evals))
				#a = sess.run(mainQN.predict,feed_dict={mainQN.stateInput:[s]})[0]
				a = np.argmax(vals - 0.035*np.log(-np.log(evals)))

			else:
				a = np.random.randint(0,N_ACTIONS)

			# if i==0 and render:
			# 	env.render()
			# 	print(a)

			s1,r,d,_ = env.step(a)

			r = process_reward(r,s1,d)
			s1 = processState(s1)

			sList[-1] = is_success(s1,d)
			total_r += r

			s = s1

			if d == True:
				print(total_r, s)
				break
		rList.append(total_r)
		# print("Episode ",str(i+1),"reward:",total_r, "success:",sList[-1])

	print("executed",str(n),"test episodes. mean reward:",np.mean(rList),"success rate:",np.mean(sList))

def generate_heatmaps(sess, mainQN, mainEN, sample=100, dims=(100,100)):
	#right_mat = np.zeros(discretize_dims)
	#left_mat = np.zeros(discretize_dims)
	#samples_mat = np.zeros(discretize_dims)

	qvals = np.zeros(dims)
	evals = np.zeros(dims)

	row_bins = get_bins(0, dims[0])
	col_bins = get_bins(1, dims[1])
	# print("samples from ",np.random.uniform(col_bins[0], col_bins[0 + 1], mean_over))
	# print("row bins", row_bins)
	# print("col bins", col_bins)
	for row in range(dims[0]):
		for col in range(dims[1]):
			sampled_states = [list(x) for x in 
								zip(np.random.uniform(row_bins[row], row_bins[row + 1], sample),
									np.random.uniform(col_bins[col], col_bins[col + 1], sample)
								)
							 ]

			es = sess.run(mainEN.predict_val,feed_dict={mainEN.stateInput:sampled_states})
			qs = sess.run(mainQN.predict_val,feed_dict={mainQN.stateInput:sampled_states})
			#samples_mat[row, col] = evals[0]
			evals[row,col] = np.mean(es)
			qvals[row,col] = np.mean(qs)
			#right_evals = evals[2]
			#left_evals = evals[0]
			#right_mat[row, col] = right_evals
			#left_mat[row, col] = left_evals

	return qvals,evals


num_episodes = 10000 #How many episodes of game environment to train network with.
max_epLength = 1000 #The max allowed length of our episode.
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
q_update_target_freq = 10000
e_update_target_freq = 1000
record_freq = 400000
y = .99 #Discount factor on the target Q-values
gamma_E = .99
if agent == COUNTER:
	gamma_E = 0
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 100000. #How many steps of training to reduce startE to endE.
discretize_dims = (100,100)

def main():
	env._max_episode_steps = max_epLength + 1
	pre_train_steps = batch_size * 3 * max_epLength #How many steps of random actions before training begins.
	record_filename = "record"
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M')
	path = "/cs/labs/oabend/borgr/ldqn_exploration/ldqn" + agent + str(st) + "/" #The path to save our model to.
	#h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
	#tau = 0.001 #Rate to update target network toward primary network
	start_temp = 0.035
	end_temp = 0.035
	
	mean_over = 100

	tf.reset_default_graph()
	mainQN = Qnetwork()
	targetQN =Qnetwork()
	
	mainEN = Enetwork()
	targetEN = Enetwork()

	init = tf.initialize_all_variables()

	saver = tf.train.Saver()

	trainables = tf.trainable_variables()

	targetOpsQ = updateTargetGraph(trainables,0)
	targetOpsE = updateTargetGraph(trainables,1)

	myBuffer = experience_buffer()

	record_states = np.zeros(discretize_dims)

	#Set the rate of random action decrease. 
	e = startE
	stepDrop = (startE - endE)/anneling_steps

	temp = start_temp
	temp_drop = (start_temp - end_temp)/anneling_steps

	#create lists to contain total rewards and steps per episode
	jList = []
	rList = []
	sList = []
	minList = []
	maxList = []
	greedyList = []
	total_steps = 0
	to_render = False

	#Make a path for our model to be saved in.
	if save_at_all and not os.path.exists(path):
		os.makedirs(path)


	with tf.Session() as sess:
		print("running", agent)
		print("\t".join(["Steps","Reward", "Success", "Max", "Min", "Temp", "Eps"]))
		sess.run(init)
		if load_model == True:
			print('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		
		updateTarget(targetOpsQ,sess) #Set the target network to be equal to the primary network.
		updateTarget(targetOpsE,sess)
		
		for i in range(num_episodes):
			
			if to_render:
				print("rendering episode", i)
			episodeBuffer = experience_buffer()
			#Reset environment and get first new observation

			sList.append(0)
			
			maxList.append(-float('inf'))
			minList.append(float('inf'))

			s = env.reset()
			s = processState(s)
			
			d = False
			rAll = 0
			num_greedy = 0
			j = 0
			#The Q-Network
			while j < max_epLength: #If the agent takes longer than max_epLength moves to reach either of the blocks, end the trial.
				maxList[-1] = max(maxList[-1],s[0])
				minList[-1] = min(minList[-1],s[0])

				if to_render:
					env.render()

				j+=1
				#Choose an action by greedily (with e chance of random action) from the Q-network
				a = action_selection(sess, s, temp, e, mainQN, mainEN)
				a_greedy = sess.run(mainQN.predict,feed_dict={mainQN.stateInput:[s]})[0]
				if a == a_greedy:
					num_greedy += 1

				record_states[discretize(s, discretize_dims)] += 1
				s1,r,d,_ = env.step(a)
				sList[-1] = is_success(s1,d) # r is positive only in success while d can be due to clipping
				# if d:
				# 	print("success: ",r,s)

				r = process_reward(r,s1,d)
				s1 = processState(s1)

				total_steps += 1

				if j>1:
					episodeBuffer.add(np.reshape(np.array([s_last,a_last,r_last,s,d_last,a]),[1,6])) #Save the experience to our episode buffer.
					if d:
						episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d,0]),[1,6])) #Save the experience to our episode buffer.


				if total_steps > pre_train_steps:
					if e > endE:
						e -= stepDrop

					if temp > end_temp:
						temp -= temp_drop
					
						
					if total_steps % update_freq == 0:
						trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
						#Below we perform the Double-DQN update to the target Q-values
						#Q1 = sess.run(mainQN.predict_val,feed_dict={mainQN.stateInput:np.vstack(trainBatch[:,3])})
						Q1 = sess.run(targetQN.predict_val,feed_dict={targetQN.stateInput:np.vstack(trainBatch[:,3])})
						#Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
						end_multiplier = -(trainBatch[:,4] - 1)
						#doubleQ = Q2[range(batch_size),Q1]
						targetQ = trainBatch[:,2] + (y*Q1 * end_multiplier)
						#Update the network with our target values.
						_ = sess.run(mainQN.updateModel, \
							feed_dict={mainQN.stateInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
						
						#print(trainBatch[:,5])
						E1 = sess.run(targetEN.out,feed_dict={targetEN.stateInput:np.vstack(trainBatch[:,3])})
						E1 = E1[range(batch_size), np.array(trainBatch[:,5], dtype=np.int32)]
						targetE = (gamma_E*E1*end_multiplier)

						_ = sess.run(mainEN.updateModel, \
							feed_dict={mainEN.stateInput:np.vstack(trainBatch[:,0]),mainEN.targetE:targetE, mainEN.actions:trainBatch[:,1]})

					if total_steps % q_update_target_freq == 0:
						updateTarget(targetOpsQ,sess) #Set the target network to be equal to the primary network.

					if total_steps % e_update_target_freq == 0:
						updateTarget(targetOpsE,sess) #Set the target network to be equal to the primary network.


					if save_at_all and total_steps % record_freq == 0:
						# compute E grids
						right_mat = np.zeros(discretize_dims)
						left_mat = np.zeros(discretize_dims)
						samples_mat = np.zeros(discretize_dims)
						row_bins = get_bins(0, discretize_dims[0])
						col_bins = get_bins(1, discretize_dims[1])
						# print("samples from ",np.random.uniform(col_bins[0], col_bins[0 + 1], mean_over))
						# print("row bins", row_bins)
						# print("col bins", col_bins)
						for row in range(discretize_dims[0]):
							for col in range(discretize_dims[1]):
								sampled_states = [list(x) for x in 
													zip(np.random.uniform(row_bins[row], row_bins[row + 1], mean_over),
														np.random.uniform(col_bins[col], col_bins[col + 1], mean_over)
													)
												 ]

								evals = sess.run(mainEN.out,feed_dict={mainEN.stateInput:sampled_states})
								samples_mat[row, col] = evals[0,0]
								evals = np.mean(evals, 0)
								right_evals = evals[2]
								left_evals = evals[0]
								right_mat[row, col] = right_evals
								left_mat[row, col] = left_evals
						# save
						# print(-np.log(left_mat))
						# print(-np.log(samples_mat))
						# print("saving to " + path+record_filename)
						np.save(path+record_filename+"_states_"+str(total_steps), record_states)
						np.save(path+record_filename+"left_evals_"+str(total_steps), left_mat)
						np.save(path+record_filename+"right_evals_"+str(total_steps), right_mat)
						np.save(path+record_filename+"_evalsnomean_"+str(total_steps), samples_mat)

				rAll += r
				s = s1
				
				s_last = s
				a_last = a
				r_last = r
				d_last = d

				if d == True:

					break
			
			myBuffer.add(episodeBuffer.buffer)
			jList.append(j)
			rList.append(rAll)
			greedyList.append(num_greedy/j)
			#Periodically save the model. 
			
			to_render = render_each and not (i%render_each)


			if save_at_all and i % save_rate == 0:
				saver.save(sess,path+'/model-'+str(i)+'.cptk')
				print("Saved Model")
			
				with open(path + "successes", "w") as fl:
					fl.writelines("\n".join([str(int(x)) for x in sList]))
				with open(path + "rewards","w") as fl:
					fl.writelines("\n".join([str(int(x)) for x in rList]))

			if len(rList) % print_rate == 0:
				output_text = [total_steps,np.mean(rList[-print_rate:]), np.mean(sList[-print_rate:]), np.mean(maxList[-print_rate:]), np.mean(minList[-print_rate:]), temp, e,  np.mean(greedyList[-print_rate:])]
				print("\t".join([str(x) for x in output_text]))
				#run_test_episode(sess,env,n=10,render = not (len(rList) % (print_rate *render_test_rate)))

		print("Percent of succesful episodes: " + str(sum(sList)/num_episodes) + "%")
		if save_at_all:	
			saver.save(sess,path+'/model-'+str(i)+'.cptk')
			print("Saved Model")
			
			with open(path + "successes", "w") as fl:
				fl.writelines("\n".join([str(int(x)) for x in sList]))
			with open(path + "rewards", "w") as fl:
				fl.writelines("\n".join([str(int(x)) for x in rList]))
	# rMat = np.resize(np.array(rList),[len(rList)//100,100])
	# rMean = np.average(rMat,1)
	# plt.plot(rMean)
	# plt.show()

if __name__ == '__main__':
	main()
