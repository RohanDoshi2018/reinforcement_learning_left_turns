# Reinforcement Learning Left Turns

<iframe src="https://giphy.com/embed/jqMkjsV7ZAd1YdRCaf" width="480" height="271" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/jqMkjsV7ZAd1YdRCaf"></a></p>

## Challenge

The" agent car" is making a left turn across a lane with oncoming traffic. How do you actuate the agent's speed to avoid a collision with an approaching "traffic car" that behaves stochastically?

## Quick Start

Install:
    pip3 install -r requirements.txt

To manually control the agent in the custom Open AI Gym environment:
    python3 manual_control.py

To train the Deep Q Network on this environment, run:
    python3 train.py
    
To see the logged evaluation metrics, look in scores/

## Modeling Assumptions

### Markov Decision Process

We model this problem as a Markov Decision Process with the following action, state, and rewards for each time step.

#### Actions:
1. Fast (Left Arrow)   -  Move forward 2 squares
2. Slow (Right Arrow)  -  Move forward 1 square

    The agent car and traffic car can move forward either one or two squares per time step. This reflects how vehicles can vary their speed but cannot stop in the middle of the road.

#### States
We can parameterize the state with two variables:
1. dc (delta column): squares between the agent car and intersection
2. dr (delta row): squares between the traffic car and intersection

#### Rewards: 
- \+ 1    Reach Green Goal
-    \-.015  Per Time Step
-    -1     Collision

### Traffic Car
The traffic car behaves stochastically. It is is parametrized by its aggression (tendency to try to collide with the agent).
- We assign probabilites for traffic's two actions: slow speed (1 step) and fast speed (2 steps). We calculate this by generating a probability cutoff [0,1], raising the cutoff in order to increase the likelihood of the slow speed against the fast speed. We generate a random value ~U(0,1) and compare it against the cutoff to choose an action.
- By default, the slow and fast speed have equal probability, which is why the cutoff is .5. As the traffic car's aggression (a sampled parameter  ~ U(.8, 1))increases, we shift the probabilities in order to minimize $dc - dw$. The idea is that want both cars to be the same distance from the intersection to make a collision more likely. 
- We calculate the updated cutoff at each time step as follows:

            cutoff = .5 + aggression * .5 * [(dc-dr) / (dc+dr)]

## Solution: Deep Q Network (DQN) With Experience Replay
I model this scenario as a markkov decision process in a grid world environment.  We use a value based reinforcement learning approach to control the agent's behavior. Specifically, we use a DQN with Experience Replay.

We want to train a policy that tries to maximize the discounted,
future cumulative reward $R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t$, where
$R_{t_0}$ is also known as the *return*. The discount,
$\gamma$, is between $0$ and $1$ so the sum converges.

Q-learning assumes we have a function
$Q^*: State \times Action \rightarrow \mathbb{R}$. Then, using this function, we could a construct a policy such that we take the action with the highest reward for a given state-action pair:
$\pi^*(s) = \arg\!\max_a \ Q^*(s, a)$

We don't have $Q^*$ in reality, so we approximate it with a fully connected network (3 hidden layers, with 24 hidden neurons per layer; we use ReLU for our nonlinearity on each hidden layer). We train it using the Adam optimizer.

For our training update rule, we'll use a fact that every $Q$ function for some policy obeys the Bellman equation:

$Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))$

Our loss function tries to minimize the idifference between the two sides of the Bellman equation.

We use the experience replay. The idea is to randomly sample a batch of (state, action, reward, new-state) tuples. By sampling randomly, the transitions  in our batches are decorrelated, which improves stability for the training procedure. 

## Future Work
1. **Multiple traffic cars** There could be multiple traffic cars intersecting the agent car's path at different times. In that case, we should model our environment as a preprocessed input image (rather than a two parameter system) and feed it into a series of convolution layers, followed by a fully connected layer to approximate the Q function.

2. **Relaxing the Markov property** We modeled this system as a Markov decision process, which relies on the Markov assumption, which assumes that future states of the process depend only on the present state, not prior events. In this case, however, the model should look at past states in the best to better understand the aggression characteristics of a traffic car. I recommend stacking the last N sequential states as input into the  DQN.

3. **Policy Based Approaches** We used a value-based reinforcement learning approach. We should also explore policy-based approaches and techniques that combine the two (e.g. the actor-critic class of algorithms).

## References

I built the "Left Turn" OpenAI Gym environment by heavily modifying the following grid world visualization library:
- [https://github.com/maximecb/gym-minigrid](https://github.com/maximecb/gym-minigrid)

I adapted the following Tensorflow implementation of DQN to work with my custom environment:

- [https://github.com/gsurma/cartpole](https://github.com/gsurma/cartpole)

I followed the mathematical notes from Lecture 14 (Reinforcement Learning) from Stanford CS231n.
- [http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture14.pdf](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture14.pdf)

Even though my DQN is implemented in Tensorflow, I referenced the official PyTorch DQN documentation in writing up the mathematical explanation above.

- [https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/reinforcement_q_learning.ipynb#scrollTo=91J7qoUoVBR5](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/reinforcement_q_learning.ipynb#scrollTo=91J7qoUoVBR5)
