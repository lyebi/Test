import numpy as np
import gym
# import universe
env=gym.make('Alien-v0')
state=env.reset()

import cv2


state=cv2.resize(state,(84,84))
print(np.shape(state))
cv2.imshow('img',state)
cv2.waitKey()