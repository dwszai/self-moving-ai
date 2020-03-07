# AI self driving class

import numpy as np 					# work with arrays
import random 						# take random sample from different samples
import os							# SAVE, LOAD model/brain
import torch 						# implement neural network
import torch.nn as nn				# contains tools of neural network
import torch.nn.functional as F 	# different functions of neural network, uber loss
import torch.optim as optim 		# optimiser for stochastic gradient descent
from torch.autograd import Variable #convert tensor into variable with gradient

# Arichitecture of the Neural network class
class Network(nn.Module):

	def __init__(self, input_size, output_size):
		super(Network, self).__init__()            # inherit function from Module
		self.input_size = input_size
		self.output_size = output_size
		self.fc1 = nn.Linear(input_size, 30)       # full connection between neurons of input and hidden layers, number of neurons can be changed
		self.fc2 = nn.Linear(30, output_size)      # full connection between neurons of hidden layers and output layer
		
	def forward(self, state):                      # function to return the q values at each state, forward the neurons
		hidden_neuron = F.relu(self.fc1(state))    # activate hidden neurons using relu method and pass in the input neurons from fc1
		q_values = self.fc2(hidden_neuron)         # return q values for each possible actions by passing in the hidden neurons
		return q_values


# Experience Replay class
#---- Create memory to store last few states for q learning to practice
class ReplayMemory(object):
	
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []                           # initialise the list of memories
		
	def push(self, event):                         # function to push this into the memory list
		self.memory.append(event)                  # add this into the memory list
		if len(self.memory) > self.capacity:       # if number of events in the memory more than the capacity it can hold,
			del self.memory[0]                     # delete the first event in the memory
	
	def sample(self, batch_size):                  # batch size is the sample size
		# if list = [(state, action, reward), (state2, action2, reward2)], then zip(*list) = [(state, state2), (action, action2), (reward, reward1)]
		samples = zip(*random.sample(self.memory, batch_size))       
		return map(lambda x: Variable(torch.cat(x, 0)), samples)    # put each batch into pytorch to get gradient


# Deep Q-learning
class Dqn():

	def __init__(self, input_size, output_size, gamma):
		self.gamma = gamma
		self.reward_window = []                            # mean of the last 100 rewards appended to the list
		self.model = Network(input_size, output_size)      # model object of neural network
		self.memory = ReplayMemory(100000)                 # number of events in the memory to get smaller sample
		self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)        # connect Adam optimizer to our NN, object of Adam class optimizer,  # to improve efficiency due to large number of arguments, learning rate can be changed
		self.last_state = torch.Tensor(input_size).unsqueeze(0)                 # create a fake dimension, first dimention of the last state at index 0
		self.last_action = 0                               # initial state action 
		self.last_reward = 0                               # between -1 and 1

	def select_action(self, state):                        # ai decide the correct action by taking the q values/output values from the state   
		# take the neural network output(q values) from the self.model object created above]
		probability = F.softmax(self.model(Variable(state))*100)     # temperature parameter = 7, the certainty of which action, higher the more certainty, T=0 means no ai
		action = probability.multinomial(1)
		return action.data[0,0]

	def learn(self, batch_state, batch_next_state, batch_reward, batch_action):               # transition/event of the states
        
		output = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)      # using gather to take the best action that we want, add unsqueeze to compensate for the fake variable created above, then squeeze to remove it after
		next_output = self.model(batch_next_state).detach().max(1)[0]                         # next output from next state, only take the max action where action is index 1, state is index 0
		target = self.gamma * next_output + batch_reward                                      # based on formula 
		td_loss = F.smooth_l1_loss(output, target)                                            # temporal difference loss -> smooth is the best loss function to improve NN
		# perform stochastic gradient and update the weights
		self.optimizer.zero_grad()                                                            # zero grad reintialize the optimzer in each loop    
		td_loss.backward()                                                                    # backpropagates the error into the NN
		self.optimizer.step()                                                                 # uses the optimzer to update the weights

	
	def update(self, reward, new_signal):                                  # update all the elements in the new transition
		new_state = torch.Tensor(new_signal).float().unsqueeze(0)          # new signal stated in map file
		self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))   # update the memory by adding new state for each transition
		action = self.select_action(new_state)                             # object of Dqn class, new action from input state is the new state (most recent)
		if len(self.memory.memory) > 100:                                  # first memory is the object of class replayMemory while 2nd memory is the attribute of the class
			batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) # returned by the sample function, 100 batch size
			self.learn(batch_state, batch_next_state, batch_reward, batch_action)
		self.last_action = action             # update the last action
		self.last_state = new_state           # update the last state
		self.last_reward = reward             # update the last reward
		self.reward_window.append(reward)     # update the reward window
		if len(self.reward_window) > 1000:    # window of fixed size 1000
			del self.reward_window[0]         # del first reward if over
		return action
	
    # mean of all the reward in the reward window
	def score(self):                          
		return sum(self.reward_window) / (len(self.reward_window) + 1)    # +1 to ensure denominator is not 0 that crashes the program
	
    # save the model
	def save(self):     
        # take the model(last version of the weights) and the optimizer that is connected to it. saved to the file named last_brain.pth
		torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
	
    # load the model    
	def load(self):   
		if os.path.isfile('last_brain.pth'):                              # if this file in working directory/ same directory as executed file
			print('loading checkpoint...')
			checkpoint = torch.load('last_brain.pth')                     # load file
			self.model.load_state_dict(checkpoint['state_dict'])          # update parameters of our model and optimizer from loaded file
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print('DONE!')
		else:
			print('No checkpoint found..')
		
		
		