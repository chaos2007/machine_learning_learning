#Importing Libararies
import gym
import numpy as np
import time
import pickle

#Environment Setup
env = gym.make("MsPacman-v0")

# Q table implementation
from collections import defaultdict
Q = defaultdict(float)
try:
    print("loading training data")
    with open("pacman_training.dat", "rb") as f:
        Q = pickle.load(f)
    print("training data loaded")
except:
    print("no training data, starting from scratch")
    Q = defaultdict(float)

gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param

actions = range(env.action_space.n)

epsilon = 0.1 #10% random
def update_Q(s, r, a, s_next, done):

    flattened_state = tuple(s.flatten())
    max_q_next = max([Q[flattened_state, a] for a in actions]) 
    # Do not include the next state's value if currently at the terminal state.
    Q[flattened_state, a.n] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[flattened_state, a.n])

def act(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    qvals = {a: Q[state, a] for a in actions}
    max_q = max(qvals.values())
    actions_with_max_q = [a for a,q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)


for episode in range(1,1001):
    done = False
    G, reward = 0,0
    state = env.reset()
    print('Episode {}'.format(episode))
    while done != True:
        action = act(tuple(state.flatten()))
        state2, reward, done, info = env.step(action)
        update_Q(state, reward, env.action_space, state2, done)
        state = state2
        if episode % 2 == 0:
            env.render()
            import time
            time.sleep(0.05)
    if episode % 100 == 0:
        with open("pacman_training_temp.dat", "wb") as f:
            print("writing training data")
            pickle.dump(Q, f, protocol=pickle.HIGHEST_PROTOCOL)
        if os.path.exists("pacman_training.dat"):
            os.remove("pacman_training.dat")
        os.rename("pacman_training_temp.dat", "pacman_training.dat")

