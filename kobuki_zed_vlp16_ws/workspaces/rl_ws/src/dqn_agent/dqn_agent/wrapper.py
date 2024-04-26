import cv2
import math
import numpy as np
from collections import deque


class GRAY_SCALE_OBSERVATION:
    def __init__(self):
        pass

    def forward(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation


class CROP_OBSERVATION:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def forward(self, observation):
        observation = observation[self.y1 : self.y2, self.x1 : self.x2]
        return observation


class RESIZE_OBSERVATION:
    def __init__(self, shape):
        self.shape = shape

    def forward(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation


class BLUR_OBSERVATION:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, observation):
        observation = cv2.GaussianBlur(observation, (self.kernel_size, self.kernel_size), 0)
        return observation


class FRAME_STACKING:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.stack = deque(maxlen=stack_size)

    def forward(self, observation):
        self.stack.append(observation)
        if len(self.stack) < self.stack_size:
            while len(self.stack) < self.stack_size:
                self.stack.append(observation)
        return np.stack(self.stack, axis=0)
