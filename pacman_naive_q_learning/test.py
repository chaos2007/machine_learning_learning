#Importing Libararies
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
import time
import pickle
from collections import defaultdict

#Environment Setup
env = gym.make("MsPacman-v0")

# Q table implementation
Q = defaultdict(float)

gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param



class DiscretizedObservationWrapper(gym.ObservationWrapper):
    """This wrapper converts a Box observation into a single integer.
    """
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.observation_space = Discrete(n_bins ** low.flatten().shape[0])

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)

actions = range(env.action_space.n)
env = DiscretizedObservationWrapper(env)
env.reset()
print("finished initializing")

def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s, a] for a in actions]) 
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a.n] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a.n])

for episode in range(1,1001):
    done = False
    G, reward = 0,0
    state = env.reset()
    print('Episode {}'.format(episode))
    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        update_Q(state, reward, env.action_space, state2, done)
        state = state2
        if episode % 10 == 0:
            env.render()
            import time
            time.sleep(0.05)
