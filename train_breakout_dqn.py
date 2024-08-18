# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:09:56 2023

@author: Enrico
"""
import gymnasium as gym
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,Lambda)
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
import random
import time
import os
import pickle 
from datetime import datetime
from config import (ENV_NAME,CLIP_REWARD,history_length,INPUT_SHAPE,BATCH_SIZE,LEARNING_RATE,MAX_EPISODE_LENGTH,FRAMES_BETWEEN_EVAL,
                    EVAL_LENGTH ,gamma,eps_initial, epsilon , eps_final,eps_final_frame,eps_evaluation, epsilon_greedy_frames, 
                    min_replay_buffer_size,max_memory_length,update_target_network_frame ,max_frames,clip_reward ,enable_priority_rb ,offset ,
                    priority_scale,SAVE_PATH,enable_from_scratch)

# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed

#import psutil
#py = psutil.Process(os.getpid())

####my ATARI breakout - v4 
def random_action():
    # Create a breakout environment
    env = gym.make('BreakoutDeterministic-v4',render_mode='human')
    # Reset it, returns the starting frame
    frame = env.reset()
    # Render
    env.render()
    is_done = False
    while not is_done:
      # Perform a random action, returns the new frame, reward and whether the game is over
      frame, reward, is_done, _,_ = env.step(env.action_space.sample())
      #cv2.imshow('ImageWindow', frame)
      #cv2.waitKey()
      # Render
      env.render()

#random_action()

###############################################################################
#### some functions
### let’s implement the image preprocessing ####################################

def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    with tf.device('/device:GPU:0'):
        frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work
    
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[34:34+160, :160]  # crop image 160 X 160 --> why cripping and not resize directly ! mybe removing not necessaty part of frame (ipper part) !
        frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
        
        #### just to visualize gray scale image cropped !
        #cv2.imshow('ImageWindow', frame)
        #cv2.waitKey()
        
        frame = frame.reshape((*shape, 1))
        #fmm= np.expand_dims(frame, axis=2) 
        
        ### to debug ##############################################################
        #dw_sampling = frame[::2, ::2] ### 105×80 note this method only allows integer resizing
        #gray_scale = np.mean(dw_sampling, axis=2).astype(np.uint8)

        return frame
###############################################################################

#### define the NN (functional) model ######################################################
###note: multiple input in NN
def game_model(n_actions, learning_rate=0.00001, input_shape=(84, 84), history_length=4):
    """Builds a (dueling) DQN as a Keras model
        Arguments:
            n_actions: Number of possible action the agent can take
            learning_rate: Learning rate
            input_shape: Shape of the preprocessed frame the model sees
            history_length: Number of historical frames the agent can see
        Returns:
            A compiled Keras model
    """

    #ATARI_SHAPE = (4, 105, 80)
    frames_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    #actions_input = Input((n_actions,), name='mask')
    
    
    normalized = Lambda(lambda x: x / 255)(frames_input)  # normalize by 255
    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = Conv2D(16, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu',use_bias=False)(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = Conv2D(64, (4, 4), strides=2,  kernel_initializer=VarianceScaling(scale=2.), activation='relu',use_bias=False)(conv_1)
    conv_3 = Conv2D(64, (3, 3), strides=1,kernel_initializer=VarianceScaling(scale=2.), activation='relu',use_bias=False)(conv_2)
    
    #x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    #x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    #x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    
    # Flattening the second convolutional layer.
    conv_flattened = Flatten()(conv_3)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = Dense(512,kernel_initializer=VarianceScaling(scale=2.), activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = Dense(n_actions,kernel_initializer=VarianceScaling(scale=2.))(hidden)
    # Finally, we multiply the output by the mask!
    model = Model(inputs=frames_input, outputs = output)
    
    #optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    #model.compile(optimizer, loss='mse')
    ### see https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26 why use hubner loss !
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
    
    return model
    
    
###############################################################################


### ring buffer class implementaiton for replay buffer ########################
### to eval if using deque is better ot not (deque is very slow)
class RingBuffer:
    def __init__(self, size=1000000):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
    def sample(self, n=20):
        l = len(self)
        return [self[int(np.random.uniform(0, 1) * l)] for _ in range(n)]

######## from gym #############################################################
class GameWrapper:
    """Wrapper for the environment provided by Gym"""
    def __init__(self, env_name, no_op_steps=10, history_length=4, render_mode=None):
        self.env = gym.make(env_name,render_mode)
        self.no_op_steps = no_op_steps
        self.history_length = 4

        self.state = None
        self.last_lives = 0
        
        self.render_mode = render_mode
        
    def reset(self, evaluation=False):
        """Resets the environment
        Arguments:
            evaluation: Set to True when the agent is being evaluated. Takes a random number of no-op steps if True.
        """

        self.frame = self.env.reset()[0] ### take the forst index this is the image matrix
        self.last_lives = 0

        # If evaluating, take a random number of no-op steps.
        # This adds an element of randomness, so that the each
        # evaluation is slightly different.
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)

    def step(self, action):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns an np.array with rgb values
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            life_lost: Whether a life has been lost
            new_frame: The raw new frame as a result of that action
            If render_mode is set to 'rgb_array' this also returns the rendered rgb_array
        """
        new_frame, reward, terminal,_, info = self.env.step(action)

        # In the commonly ignored 'info' or 'meta' data returned by env.step
        # we can get information such as the number of lives the agent has.

        # We use this here to find out when the agent loses a life, and
        # if so, we set life_lost to True.

        # We use life_lost to force the agent to start the game
        # and not sit around doing nothing.
        if info['lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['lives']

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2) ## adding new processed frame at the end removing the first

        if self.render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(self.render_mode)
        elif self.render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, life_lost

#######replay buffer class
class ReplayBuffer:
    """Replay Buffer to store transitions"""
    def __init__(self, size=100000, input_shape=(84, 84), history_length=4):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
            use_per: Use PER instead of classic experience replay
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)


    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        if clip_reward:
            reward = np.sign(reward) ### -1 or +1

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        """Returns a minibatch of self.batch_size = 32 transitions
        """

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities based on priority weights
                index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-self.history_length:idx, ...])
            new_states.append(self.frames[idx-self.history_length+1:idx+1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))
        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]



###############################################################################

##### class agent definition 
class Agent(object):
    """Implements a standard DDDQN agent"""
    def __init__(self,
                 dqn,
                 target_dqn,
                 n_actions,
                 input_shape=(84, 84),
                 batch_size=32,
                 history_length=4,
                 ):
             
         # DQN
         self.DQN = dqn
         self.target_dqn = target_dqn
         self.n_actions = n_actions
         self.input_shape = input_shape
         self.batch_size = batch_size
         self.history_length = history_length


    
    def get_action(self, state, eps):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for
        Returns:
            An integer as the predicted move
        """

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        # Otherwise, query the DQN for an action
        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)),verbose = 0)[0]
        return q_vals.argmax()


    def update_target_network(self):
         """Update the target Q network"""
         self.target_dqn.set_weights(self.DQN.get_weights())
         # SAVE_PATH = 'C:/Users/Enrico/Desktop/Progetti/9 REIN_FLEARNING/Deep-Q-Learning_Atari_game/my_data_saved'
         # self.target_dqn.save( SAVE_PATH  + '/target_dqn.h5')
         # self.DQN.save( SAVE_PATH  + '/dqn.h5')
         
         
    def learn(self,states,new_states,rewards_sample,action_sample,num_actions, gamma,terminal_flags_sample ):
        # Main DQN estimates best action in new states
        #arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

        # Target DQN estimates q-vals for new states
        #future_q_vals = self.target_dqn.predict(new_states)
        future_rewards = self.target_dqn.predict(new_states,verbose = 0)
        # Calculate targets (bellman equation)
        updated_q_values = rewards_sample + (gamma * (tf.reduce_max(future_rewards, axis=1))*(1-terminal_flags_sample)) ### calc Terget temporal difference !
        
        #target_q = rewards + (gamma*double_q * (1-terminal_flags))
        
        # Create a mask so we only calculate loss on the updated Q-values
        #masks = tf.one_hot(action_sample, num_actions) ## create one hot encoder matrix using action used and multiply q_values estimate from main NN 

        # Use targets to calculate loss (and use loss to calculate gradients)
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(action_sample, num_actions, dtype=np.float32)  # using tf.one_hot causes strange errors
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)
            
            #Q = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)### create a vector of 32 positions

            error = Q - updated_q_values
            loss = tf.keras.losses.Huber()(updated_q_values, Q)

            # if self.use_per:
            #     # Multiply the loss by importance, so that the gradient is also scaled.
            #     # The importance scale reduces bias against situataions that are sampled
            #     # more frequently.
            #     loss = tf.reduce_mean(loss)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        # if self.use_per:
        #     self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error
    
    def save(self, folder_name,experience_list, **kwargs):
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')
        
        # Open a file and use dump() 
        with open(folder_name+'/repl_buffer.pkl', 'wb') as file: 
            # A new file will be created 
            pickle.dump(experience_list, file) 
            
    def load(self, folder_name):

            if not os.path.isdir(folder_name):
                raise ValueError(f'{folder_name} is not a valid directory')

            # Load DQNs
            self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
            self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
            self.optimizer = self.DQN.optimizer
            
            ### load pickle replay buffer 
            # Open the file in binary mode 
            with open(folder_name+'/repl_buffer.pkl', 'rb') as file: 
                # Call load method to deserialze 
                mypickle = pickle.load(file) 
            return mypickle
        

###############################################################################
if __name__ == "__main__":
    ### var init ###############################################################
    ### epsilon decay definition--> define two linear decays 
    slope = -(eps_initial - eps_final) / epsilon_greedy_frames
    intercept = eps_initial - slope*min_replay_buffer_size
    slope_2 = -(eps_final - eps_final_frame) / (max_frames - epsilon_greedy_frames - min_replay_buffer_size)
    intercept_2 = eps_final_frame - slope_2*max_frames
    ###loss
    loss_list = []
    episode_reward_history = []

    ##### inic class ###########################################################
    game_wrapper = GameWrapper(ENV_NAME)
    # Build main and target networks
    MAIN_DQN = game_model(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
    TARGET_DQN = game_model(game_wrapper.env.action_space.n,LEARNING_RATE, input_shape=INPUT_SHAPE)
    ### call clss agent
    agent = Agent(MAIN_DQN, TARGET_DQN, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, history_length=4)
    
    #reward_for_saving = 70

    
    episode_count = 0
    # Training and evaluation #################################################
    if  enable_from_scratch: ### start from beginning
        frame_number = 0
        count = 0  # total index of memory written to, always less than size
        current = 0  # index to write in replay buffer
        episode_count = 0
        rewards = []
        loss_list = []
        episode_reward_history = []
        ### replay buffer setting vars
        # Write memory
        # Pre-allocate memory replay buffer ###################################
        actions = np.empty(max_memory_length, dtype=np.int32)
        rewards = np.empty(max_memory_length, dtype=np.float32)
        frames = np.empty((max_memory_length, INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.uint8)
        terminal_flags = np.empty(max_memory_length, dtype= bool)
        priorities = np.zeros(max_memory_length, dtype=np.float32)
    else: # use saved data in selected folder 
        replay_buffer_pickle= agent.load(SAVE_PATH)
        actions =  replay_buffer_pickle[0]
        rewards= replay_buffer_pickle[1]
        terminal_flags = replay_buffer_pickle[2]
        count = replay_buffer_pickle[3]
        current = replay_buffer_pickle[4]
        frame_number = replay_buffer_pickle[5]
        epsilon= replay_buffer_pickle[6]
        frames = replay_buffer_pickle[7]
        ###to work on this
        #priorities = np.zeros(max_memory_length, dtype=np.float32)
        priorities = replay_buffer_pickle[8]
    ###########################################################################
    try:
        while True:  # Run until solved
              epoch_frame = 0
              while epoch_frame < FRAMES_BETWEEN_EVAL:
                  start_time = datetime.now() #time.time()
                  game_wrapper.reset()
                  life_lost = True
                  episode_reward_sum = 0
                  # print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))
                  # print ('RAM perc: {}'.format(py.cpu_percent(interval=0.5)))
                  # print ('Virtual mem perc: {}'.format(psutil.virtual_memory().percent))
                  for _ in range(MAX_EPISODE_LENGTH):
                        # Get action
                        action = agent.get_action(game_wrapper.state, epsilon) #frame_number, state, evaluation, eps
                        #### after action update epsilon according to greedy approach #####
                        if frame_number < min_replay_buffer_size:
                            epsilon =  eps_initial
                        elif frame_number >=  min_replay_buffer_size and frame_number < min_replay_buffer_size + epsilon_greedy_frames:
                            epsilon =  slope*frame_number + intercept
                        elif frame_number >= min_replay_buffer_size + epsilon_greedy_frames:
                            epsilon = slope_2*frame_number + intercept_2
                        ###################################################################
                        
                        # Take step using action selected from greedy plocy
                        processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                        frame_number += 1 ### total numbe rof frames 
                        epoch_frame += 1 ### numper of epoch done per episode before terminal !
                        episode_reward_sum += reward
                        
                        ##### updart replay buffer adding experience ######################
                        if clip_reward:
                            reward = np.sign(reward) ### -1 or +1 or 0
            
                        ### just take [84X84] and fill the replay buffer
                        frame=processed_frame[:, :, 0]
            
                        # Write memory (replay buffer)#####################################
                        actions[current] = action
                        frames[current, ...] = frame
                        rewards[current] = reward
                        terminal_flags[current] = life_lost
                        priorities[current] = max(priorities.max(), 1)  # make the most recent experience important
                        count = max(count, current+1) ### counting replay buffer --> # total index of memory written to, always less than self.size
                        current = (current + 1) % max_memory_length ### when arrives at the and start writing oldest index == 0 
                        ####################################################################
                        # Update agent 
                        if frame_number % history_length== 0 and count> min_replay_buffer_size : 
                            ##### get minibatch
                            if count < history_length :
                                raise ValueError('Not enough memories to get a minibatch')
            
                            # Get a list of valid indices #################################
                            if enable_priority_rb:
                                #scaled_priorities = priorities[0:count-1] ** priority_scale
                                scaled_priorities = priorities[history_length:count-1] ** priority_scale #priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
                                sample_probabilities = scaled_priorities / sum(scaled_priorities)
                            
                            
                            indices = []
                            for i in range(BATCH_SIZE):
                                while True:
                                    # Get a random number from history_length to maximum frame written 
                                    if(enable_priority_rb):
                                        #index = np.random.choice(np.arange(0, count - 1), p=sample_probabilities)
                                        index = np.random.choice(np.arange(history_length, count - 1), p=sample_probabilities)
                                    else:
                                        #index = random.randint(0, count - 1)
                                        index = random.randint(history_length, count - 1)
            
                                    # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                                    if index >= current and index - history_length <= current:
                                        continue
                                    if terminal_flags[index - history_length:index].any(): ##### if some state btw hystory (4) is terminal state not valid index
                                        continue
                                    break
                                indices.append(index)
                                
                            # Retrieve states from memory using indices ###################
                            states = []
                            new_states = []
                            for idx in indices:
                                states.append(frames[idx-history_length:idx, ...]) # -> now
                                new_states.append(frames[idx-history_length+1:idx+1, ...]) #--> next 
            
                            states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
                            new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))
                            action_sample = actions[indices]
                            rewards_sample = rewards[indices]
                            terminal_flags_sample = terminal_flags[indices]
                            
                            loss, errors = agent.learn(states, new_states, rewards_sample, action_sample, history_length, gamma,terminal_flags_sample )
                            loss_list.append(loss)
                            #if(enable_priority_rb):
                            for i, e in zip(indices, errors):
                                    priorities[i] = abs(e) + offset
            
                                
                            
                        # Update target network every 1000 frames after frame greater 50000 ( update_target_network_frame )
                        if  frame_number % update_target_network_frame==0 and frame_number > min_replay_buffer_size: 
                            agent.update_target_network()
            
                        # Break the for loop when the game is over
                        if terminal:
                            terminal = False
                            break
                 #### out for loop ############################################
                  ### add data to list
                  episode_count += 1
                  episode_reward_history.append(episode_reward_sum)
                  if len(episode_reward_history) % 10 == 0: ## printing info 
                    end_time = datetime.now()
                    difference = end_time - start_time 
                    seconds = difference.total_seconds() 
                    print(f'Game number: {str(len(episode_reward_history)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(episode_reward_history[-10:]):0.1f} Epoch frame eval: {epoch_frame} epsilon: {epsilon}  Time taken: {seconds :.1f}s')

              # Evaluation every `FRAMES_BETWEEN_EVAL` frames
              terminal = True
              eval_rewards = []
              evaluate_frame_number = 0
              for _ in range(EVAL_LENGTH): #### looping frames 
                  if terminal:
                      game_wrapper.reset(evaluation=True)
                      life_lost = True
                      episode_reward_sum = 0
                      terminal = False ### reset terminal
                  # Breakout require a "fire" action (action #1) to start the game each time a life is lost.
                  if life_lost :
                        action = 1 
                  else:
                        action = agent.get_action(game_wrapper.state,epsilon)
                  
                  # Step action
                  processed_frame, reward, terminal, life_lost = game_wrapper.step(action)

                  episode_reward_sum += reward
                  
                  # On game-over
                  if terminal:
                      eval_rewards.append(episode_reward_sum)
              if len(eval_rewards) > 0:
                final_score = np.mean(eval_rewards)
              else:
                # In case the game is longer than the number of frames allowed
                final_score = episode_reward_sum
            
              # Save model ####################################################
              print('Evaluation score:', final_score)
              if(final_score>400):
                  print('Saving...')
                  experience_list = [actions,rewards,terminal_flags,count, current, frame_number, epsilon, frames, priorities ] #### add priorities
                  agent.save(SAVE_PATH,experience_list)
                  print('Saved.')
                  break
                  
              
              '''if len(episode_reward_history) > 80 and SAVE_PATH is not None:
                  experience_list = [actions,rewards,terminal_flags,count, current, frame_number, epsilon, frames, priorities ] #### add priorities
                  agent.save(SAVE_PATH,experience_list)
                  break'''
                    
              '''running_reward = np.mean(episode_reward_history[-100:]) ### get last 100 values !
              if running_reward >= 380:  # Condition to consider the task solved
                  print("Solved at episode {}!".format(episode_count))
                  experience_list = [actions,rewards,terminal_flags,count, current, frame_number, epsilon, frames, priorities ] #### add priorities
                  print('Saving...')
                  agent.save(SAVE_PATH,experience_list)
                  print('Saved.')
                  break'''
                    
              if(frame_number>= max_frames):
                  print("Solved at frame {}!".format(frame_number))
                  print('Saving...')
                  experience_list = [actions,rewards,terminal_flags,count, current, frame_number, epsilon, frames, priorities ] #### add priorities
                  agent.save(SAVE_PATH,experience_list)
                  print('Saved.')
                  break
              
    except KeyboardInterrupt:
         print('\nTraining exited early.')

         if SAVE_PATH is None:
             try:
                 SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
             except KeyboardInterrupt:
                 print('\nExiting...')
        
         if SAVE_PATH is not None:
            print('Saving...')
            experience_list = [actions,rewards,terminal_flags,count, current, frame_number, epsilon, frames, priorities ] #### add priorities
            agent.save(SAVE_PATH,experience_list)
            print('Saved.')


