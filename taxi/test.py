#Importing Libararies
import gym
import numpy as np

#Environment Setup
env = gym.make("Taxi-v2")
env.reset()

# Q table implementation
Q = np.zeros([env.observation_space.n, env.action_space.n])
G = 0
gamma = 0.618
for episode in range(1,1001):
    done = False
    G, reward, counter = 0,0,0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state,action] = (reward + gamma * np.max(Q[state2]))
        G += reward
        counter += 1
        state = state2
        if episode % 50 == 0:
            print('\033[H\033[J')
            env.render()
            print('Episode {} Total Reward: {} counter: {}'.format(episode,G,counter))
            import time
            time.sleep(0.1)

for episode in range(1,2):
    done = False
    G, reward, counter = 0,0,0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state,action] = (reward + gamma * np.max(Q[state2]))
        G += reward
        counter += 1
        state = state2   
        print('\033[H\033[J')
        env.render()
        import time
        time.sleep(0.1)

