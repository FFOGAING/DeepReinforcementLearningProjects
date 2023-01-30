# Banana Collector

![image1](image/banana.gif)

The goal of this project is to train an agent to navigate in a large square world and collects yellow bananas. The simulation environment is provided by [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).
This project was solved with a **Double Deep Q-Networks (DDQN)** algorithm.

## Overview
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the environment is considered as solved if the agent get an average score of +13 over 100 consecutive episodes.

## Getting started

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name name_of_your_env python=3.6
	source activate name_of_your_env
	```
	- __Windows__: 
	```bash
	conda create --name name_of_your_env python=3.6 
	activate name_of_your_env
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	
3. Clone the repository and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/FFOGAING/DeepReinforcementLearningProjects.git
cd DeepReinforcementLearningProjects/Banana_collector/python
pip install .
```
4. If you encounter a problem with the pytorch versions, follow this line
```bash
pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `name_of_your_env` environment.  
```bash
python -m ipykernel install --user --name name_of_your_env --display-name "name_of_your_env"
```
7. Build your own unity environemnt from the [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) other dowload a already builded one with one of theses Links
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

 Then, place the file in the `Banana_collector/` folder of the repository, and unzip (or decompress) the file.

8. Run the `main` python file either in anaconda other in an editor of your choice (like Visual studio Code other Pycharm). Make sure to activate the virtual environment other to select the right interpreter  

	- __Anaconda__: 
	```bash
	python3 main
	
	```
	- __Visual Studio Code__: 
	  - `ctrl + shift + p` to start the  **Command Palette**
 	  - write and select the `python: select interpreter`
	  - choose the correct interpreter

## Future Works
### Extensions
Some extensions of the Double Deep-Q-Networks could be implement to reach better performance 
- Prioritized DDQN
- Dueling DDQN
- Distributional DDQN
### Next Challenge: Learning from Pixels
For this challenge a new Unity environement is needed
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

The agent schould learn this time  directly from pixels. This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view
## Related Papers
- [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 
- [Issues in Using Function Approximation for Reinforcement Learning](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)