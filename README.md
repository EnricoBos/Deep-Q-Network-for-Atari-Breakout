# Reinforcement Learning: Deeq Q Learning Network (DQN) Agent playing Atari Breakout

* Implementation in Tensorflow of Deep Q Learning Network (DQN) in Atari's Breakout environment, .


## Environment
* **< Python 3.10.10 >**
* **< [OpenAI Gymnasium] >**
	- Install the OpenAI Gymnasium (V.0.28.1) Atari environment:
	 'pip install gymnasium'
	- Atari environment used: `BreakoutNoFrameskip-v4`
* **< [Tensorflow V.2.10.0](https://www.tensorflow.org/) >**

## Implementation
* Deep Q Learning Network with the following improvements:
	- **Experience Replay**
	- **Fixed Target Q-Network**
	- **TD error loss function** with: *Q<sub>target</sub> = reward + (1-terminal) * (gamma * Q<sub>max</sub>(s’)
)*


## Usage (be sure to define a file save path)
* Traing the DQN Agent: `my_breakout_dqn.py`
* Testing the DQN Agent: `my_breakout_dqn_eval.py`


## Learning Curve
* Single learning curve: