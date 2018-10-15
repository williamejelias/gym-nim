import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class NimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.numberOfHeaps = 4
        self.maxHeapSize = 50
        self.heaps = None
        self.state = None
        self.generateActionSpace()
        self.setHeapsStartingPositions()
        print("Initialised")
        return

    def setMaxHeapSize(self, size=30):
        self.maxHeapSize = size
        self.generateActionSpace()
        # regenerate action space
        return None

    def setNumberOfHeaps(self, number=4):
        self.numberOfHeaps = number
        # regenerate action space
        self.generateActionSpace()
        return None

    def setHeapsStartingPositions(self, heaps=None):
        if heaps:
            self.heaps = np.array(heaps)
            currentMax = np.max(self.heaps)
            if self.maxHeapSize < currentMax:
                self.setMaxHeapSize(currentMax)

        # the starting position are random-sized heaps of beans
        else:
            self.heaps = np.random.randint(low=1, high=self.maxHeapSize + 1, size=(self.numberOfHeaps,))

        self.reset()
        return None

    def step(self, action):
        reward = 0
        done = False

        # Action should be a list of size 2
        assert isinstance(action, list), 'Wrong type.' + \
                                         "Type is: {}. Should be <class 'list'>".format(type(action))

        assert len(action) == 2, 'Wrong length.' + \
                                 'Length is:{}. Should by 2'.format(len(action))

        heapNumber = action[0]
        numberOfBeans = action[1]
        heapSize = self.state[heapNumber]

        assert heapSize >= numberOfBeans, 'Heap is not big enough' + \
                                          "Heap number is: {}. You tried to take {}, but there are only {} beans in " \
                                          "this heap. These are the heaps: {}".format(
                                              heapNumber, numberOfBeans, heapSize, self.state)

        self.state[heapNumber] -= numberOfBeans

        # Game is over if all heaps are empty
        if np.all(np.array(self.state) == 0):
            reward = 1  # this is for the normal variant. The misère variant the reward = - 1.
            done = True

        print("Stepped")
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.heaps
        print('Reset to state initial state.')
        self.render()
        return self.state

    def render(self, mode='human'):
        for index in range(len(self.state)):
            print("Heap ", index, ": ", self.state[index])
        return None

    def generateActionSpace(self):
        # Action space is two numbers
        # First is heap number, Second is number to remove
        self.action_space = spaces.Box(np.array([0, 1]), np.array([self.numberOfHeaps, self.maxHeapSize]))
        return None
