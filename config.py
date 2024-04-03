# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:26:22 2024

@author: Enrico
"""

## the config #################################################################
ENV_NAME = 'BreakoutDeterministic-v4'
CLIP_REWARD = True                # Any positive reward is +1, and negative reward is -1, 0 is unchanged
#DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
#MIN_REPLAY_BUFFER_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent
#MEM_SIZE = 1000000                # The maximum size of the replay buffer
#MAX_NOOP_STEPS = 20               # Randomly perform this number of actions before every evaluation to give it an element of randomness
#update_after_actions  = 4
history_length=4                   # Number of actions between gradient descent steps
INPUT_SHAPE = (84, 84)             # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 32                    # Number of samples the agent learns from at once
#batch_size =  32
LEARNING_RATE = 0.00001
MAX_EPISODE_LENGTH = 30000        # Maximum length of an episode (in frames). 
#LOAD_FROM = None
FRAMES_BETWEEN_EVAL = 100000      # Number of frames between evaluations
EVAL_LENGTH = 20000               # Number of frames to evaluate for
gamma = 0.99 
####### epsilon decay vars definition
eps_initial= 1
epsilon = 1
eps_final=0.1
eps_final_frame=0.01
eps_evaluation=0.0
epsilon_greedy_frames = 1000000.0
##### replay buffer size #################################################
min_replay_buffer_size = 50000 ## this is min buffer size --> after this start bith exploring and exploiting uodating epsiolon
max_memory_length = 100000 ### max dimension of raplay buffer
###########################################################################
# How often to update the target network
update_target_network_frame = 2500
###############################################################################
max_frames=25000000
clip_reward = True
### priory var definition #####################################################
enable_priority_rb = False
offset = 0.1
priority_scale = 0.6 ####0.7
###
SAVE_PATH = 'your dir' ### change this according to folder name
###
enable_from_scratch = False ### yes--> starting learning from beginning, false ---> use saved tensors

