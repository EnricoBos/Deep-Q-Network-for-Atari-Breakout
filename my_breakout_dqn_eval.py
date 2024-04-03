# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:37:55 2023

@author: Enrico
"""

#### my eval dqn ##############################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from my_breakout_dqn import (Agent,game_model)
import gymnasium as gym
import random
import cv2
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.utils.save_video import save_video

###############################################################################
def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34+160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    #fmm= np.expand_dims(frame, axis=2) 

    return frame


class GameWrapper:
    """Wrapper for the environment provided by Gym"""
    def __init__(self, env_name, no_op_steps=10, history_length=4, render_mode=None):
        self.env = gym.make(env_name,render_mode= render_mode)
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
            #cv2.imshow('frameliveslost', new_frame)
            #cv2.waitKey()
        else:
            life_lost = terminal
        self.last_lives = info['lives']

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2) ## adding new processed frame at the end removing the first

        if self.render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(self.render_mode)
        elif self.render_mode == 'human':
            self.env.render()

        return processed_frame, reward, terminal, life_lost, new_frame


###vars
# Change this to the path of the model you would like to use
RESTORE_PATH = None
ENV_NAME = 'BreakoutDeterministic-v4'
MAX_NOOP_STEPS = 20 # Randomly perform this number of actions before every evaluation to give it an element of randomness
INPUT_SHAPE = (84, 84)# Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = batch_size =  32 # Number of samples the agent learns from at once
LEARNING_RATE = 0.00001
EVAL_LENGTH = 10000  # Number of frames to evaluate for
eps_evaluation=0.0

if RESTORE_PATH is None:
    raise UserWarning('Please change the variable `RESTORE_PATH` to where you would like to load the model from')
    
# call class env ##############################################################
game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS,history_length=4, render_mode='human')

video_path = "C:/Users/Enrico/Desktop/Progetti/9 REIN_FLEARNING/Deep-Q-Learning_Atari_game/videos/breakout_video.mp4"

print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

# init class 
MAIN_DQN = game_model(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = game_model(game_wrapper.env.action_space.n,LEARNING_RATE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, history_length=4)
### load saved models
_=agent.load(RESTORE_PATH)

print('Models Loaded')

terminal = True
eval_rewards = []
frames = []  # List to store frames
step_starting_index = 0
episode_index = 0
enable_saving = False
for frame in range(EVAL_LENGTH): #### looping frames 
    if terminal:
        game_wrapper.reset(evaluation=True)
        life_lost = True
        episode_reward_sum = 0
        terminal = False ### reset terminal
        
    # Breakout require a "fire" action (action #1) to start the
    # game each time a life is lost.
    # Otherwise, the agent would sit around doing nothing.

    #breakpoint()
    if life_lost :
        action = 1 
    else:
        action = agent.get_action(game_wrapper.state, eps_evaluation)

    # Step action
    processed_frame, reward, terminal, life_lost, not_proc_frame = game_wrapper.step(action)
    frames.append(not_proc_frame)
    episode_reward_sum += reward

    # On game-over
    if terminal:
        print(f'Game over, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
        eval_rewards.append(episode_reward_sum)
        ### frames for vide recording #########################################
        if(enable_saving): 
            save_video(frames,'meo',        
                    step_starting_index=step_starting_index,
                    episode_index=episode_index,
                    fps=29)
        step_starting_index = frame  + 1
        episode_index += 1
        break
    
#game_wrapper.env.(video_path , video_path, fps=30)
game_wrapper.env.close()
print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)

###############################################################################