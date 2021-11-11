# Reacher environment solved with Deep Deterministic Policy Gradients
This repository offers you a DDPG implementation that solves the reacher environment from the Unity ML-Agents toolkit.

There are two versions of this environment available. The first version has a single arm, while the second version has 
20 arms with their own environment copy. The results shown in this repository corresponds to the second version of the 
environment. However, the implementation should be also valid for the first version.

The environment consists on a (set of) double-jointed arm that has to reach a target location. The agent receives a
reward of +0.1 each time step the arm keeps its position at the target location. The longest the agent is able to maintain
this position, the greater is the reward.

The feature or state space has 33 dimensions containing the position, the rotation, the velocity and the angular 
velocity of the arm. On the other side, the action spaces is an array with four numbers that corresponds to the torque 
to be applied on each joint. The values in the action array are floats in the (-1, 1) range.

The environment is considered to be solved after obtaining an average score greater than 30 over 100 episodes.


# Installation

## 1: Install project dependencies
Please install the pip dependencies with the following command:

<code>pip install -r requirements.txt</code>

## 2: Download the Reacher Unity Environment
Download the version corresponding to your operative system and place the uncompressed content in the root path of this
repository:

### Version 1
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Version 2
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

# Running the code
We prepared an evaluation script <code>evaluation.py</code> which trains and produced the weights DDPG model:

You should provide the reacher executable path to the evaluation script like as follows:

<code>python evaluation.py --reacher_executable Reacher_Windows_x86_64/Reacher.exe</code>

Executing the evaluation script will generate the following files in the root folder of this repository:
* A PDF named <code>ddpg_scores.pdf</code> containing a plot of the scores achieved by each model for each timestep.
* A file called <code>checkpoint_actor.pth</code> containing the weights of the actor neural network.
* A file called <code>checkpoint_critic.pth</code> containing the weights of the critic neural network.
