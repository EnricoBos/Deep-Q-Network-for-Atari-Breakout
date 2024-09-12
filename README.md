# Reinforcement Learning: Deeq Q Learning Network (DQN) Agent playing Atari Breakout

* This repository contains an implementation of a Deep Q-Learning Network (DQN) for playing Atari's Breakout using TensorFlow. The project is designed to demonstrate how reinforcement learning techniques can be applied to solve classic arcade games.
  

## Environment Setup
Ensure your environment is properly configured with the following dependencies:
* Python 3.10.10 
* OpenAI Gymnasium: v0.28.1
* Atari Environment: BreakoutNoFrameskip-v4
* Tensorflow V.2.10.0


## Implementation
* Deep Q Learning Network with the following improvements:
	- **Experience Replay**: Storing past experiences in a replay buffer to break the correlation between consecutive samples. This allows the agent to learn from a broader distribution of experiences, improving convergence and stability.
	- **Fixed Target Q-Network**: Stabilizing the learning process by introducing a separate target network. This target network is updated less frequently than the main network, preventing oscillations and divergence during training.
	- **TD Error Loss Function**: Using Temporal Difference (TD) learning to minimize the error between predicted Q-values and target Q-values. The loss is optimized using the formula:
	
	  *Q<sub>target</sub> = reward + (1 - terminal) * (gamma * Q<sub>max</sub>(s’))*
	  
	  where:
	  - `reward`: Immediate reward received after taking an action.
	  - `terminal`: A boolean value that indicates whether the episode has ended.
	  - `gamma`: The discount factor that weighs future rewards.
	  - `Q<sub>max</sub>(s')`: The maximum predicted Q-value for the next state *s'*. This helps the agent learn to prioritize actions that lead to higher future rewards.

	  Additionally, the Bellman equation is used to update Q-values, ensuring that the agent progressively refines its policy.


## Project Structure
* my_breakout_dqn.py: Script to train the DQN agent.
* my_breakout_dqn_eval.py: Script to test and evaluate the trained DQN agent.


## Learning Curve
![learning_curve_atari_dqn_breakout](https://github.com/EnricoBos/dqn_Atari_breakout/assets/44166692/47ed29aa-b58e-45f8-b7ad-f83f553d015c)


## Result
https://github.com/EnricoBos/dqn_Atari_breakout/assets/44166692/1f8dd0d7-81d4-495d-8969-384cb2a0a5ab


## Authors
* Enrico Boscolo
