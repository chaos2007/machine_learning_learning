#Importing Libararies
import gym
import numpy as np
import time

#Environment Setup
env = gym.make("MsPacman-v0")

#Random Movements
state = env.reset()
counter = 0
done = None
while done != True:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1
    env.render()
    time.sleep(0.05)

