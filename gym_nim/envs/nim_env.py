import functools
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy


class NimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.numberOfHeaps = 4
        self.maxHeapSize = 50
        self.heaps = None
        self.state_size = self.numberOfHeaps
        self.resetState = None
        self.state = None
        self.action_size = 0
        self.outputToActionMap = []
        self.setHeapsStartingPositions()
        self.generateActionSpace()
        self.generateOutputToActionMap()
        return

    def setMaxHeapSize(self, size=30):
        self.maxHeapSize = size
        self.generateActionSpace()
        # regenerate action space
        return None

    def setNumberOfHeaps(self, number=4):
        self.numberOfHeaps = number
        self.state_size = self.numberOfHeaps
        # regenerate action space
        self.generateActionSpace()
        return None

    def setHeapsStartingPositions(self, heaps=None):
        if heaps:
            self.heaps = np.array(heaps, dtype=np.int32)
            currentMax = np.max(self.heaps)
            if self.maxHeapSize < currentMax:
                self.setMaxHeapSize(currentMax)

        # the starting position are random-sized heaps of beans
        else:
            self.heaps = np.random.randint(low=1, high=self.maxHeapSize + 1, size=(self.numberOfHeaps,))
        self.resetState = copy.deepcopy(self.heaps)
        self.generateActionSpace()
        self.reset()
        return None

    def getActionSize(self):
        sum = 0
        for heap in self.heaps:
            sum += heap
        self.action_size = sum
        return sum

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
            reward = 1  # this is for the normal variant.
            done = True

        return self.state, reward, done, {}

    def reset(self):
        self.state = copy.deepcopy(self.resetState)
        # print('Reset to state initial state.')
        # self.render()
        return self.state

    def render(self, mode='human'):
        for index in range(len(self.state)):
            print("Heap ", index, ": ", self.state[index])
        return None

    def generateActionSpace(self):
        # Action space is two numbers
        # First is heap number, Second is number to remove
        self.action_space = spaces.Box(np.array([0, 1]), np.array([self.numberOfHeaps, self.maxHeapSize]))
        self.getActionSize()
        self.generateOutputToActionMap()
        return self.action_space

    def generateOutputToActionMap(self):
        action_map = []
        for heap in range(len(self.heaps)):
            for bean in range(1, self.heaps[heap]+1):
                action_map.append([heap, bean])
        self.outputToActionMap = action_map
        pass

    def lookupAction(self, output):
        assert output <= len(self.outputToActionMap), 'Outside of range of possible actions.' + \
                                 'Output is:{}. Should be in range 0-{}'.format(output, len(self.outputToActionMap))
        return self.outputToActionMap[output]

    def getMoveList(self):
        return self.outputToActionMap

    def getPossibleMoves(self):
        pmoves = []
        for heap in range(len(self.state)):
            for bean in range(1, self.state[heap]+1):
                pmoves.append([heap, bean])
        return pmoves

    def getPossibleMoveIndices(self):
        pmoves = [self.outputToActionMap.index(item) for item in self.outputToActionMap if item in self.getPossibleMoves()]
        return pmoves

    def getIllegalMoveIndices(self):
        impmoves = [self.outputToActionMap.index(item) for item in self.outputToActionMap if item not in self.getPossibleMoves()]
        return impmoves

    def getOptimalMoves(self):
        optimal_moves = []

        # first choose heap with non-zero size
        state = self.state.copy()
        # print(state)
        nim_sum = functools.reduce(lambda x, y: x ^ y, state)

        # Calc which move to make
        for index, heap in enumerate(state):
            target_size = heap ^ nim_sum
            if target_size < heap:
                beans_number = heap - target_size
                optimal_moves.append([index, beans_number])

        return optimal_moves
