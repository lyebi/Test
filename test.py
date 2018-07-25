import numpy as np
import gym
# import universe
env=gym.make('MontezumaRevenge-v0')
state=env.reset()

import cv2





for i in range(10000):
    for a in range(18):
        for b in range(10):
            state,reward,_,_=env.step(a)
            print('reward:',reward)
            state = cv2.resize(state, (800, 800))
            cv2.imshow('img',np.uint8(state))
            cv2.waitKey()