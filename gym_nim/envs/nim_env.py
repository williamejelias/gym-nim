import copy
import functools
import gym
import numpy as np
from gym import spaces


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
        self.set_heaps_starting_positions()
        self.generate_action_space()
        self.generate_output_to_action_map()

        # Reward scheme
        self.win_reward = 1
        self.loss_reward = -1
        self.transition_reward = 0
        self.invalid_move_reward = -10
        return

    def set_max_heap_size(self, size=30):
        self.maxHeapSize = size
        self.generate_action_space()
        # regenerate action space
        return None

    def set_number_of_heaps(self, number=4):
        self.numberOfHeaps = number
        self.state_size = self.numberOfHeaps
        # regenerate action space
        self.generate_action_space()
        return None

    def set_heaps_starting_positions(self, heaps=None):
        if heaps:
            self.heaps = np.array(heaps, dtype=np.int32)
            current_max = np.max(self.heaps)
            if self.maxHeapSize < current_max:
                self.set_max_heap_size(current_max)

        # the starting position are random-sized heaps of beans
        else:
            self.heaps = np.random.randint(low=1, high=self.maxHeapSize + 1, size=(self.numberOfHeaps,))
        self.resetState = copy.deepcopy(self.heaps)
        self.generate_action_space()
        self.reset()
        return None

    def get_action_size(self):
        action_sum = 0
        for heap in self.heaps:
            action_sum += heap
        self.action_size = action_sum
        return action_sum

    def step(self, action):
        reward = 0
        done = False

        # Action should be a list of size 2
        assert isinstance(action, list), 'Wrong type.' + \
                                         "Type is: {}. Should be <class 'list'>".format(type(action))

        assert len(action) == 2, 'Wrong length.' + \
                                 'Length is:{}. Should by 2'.format(len(action))

        heap_number = action[0]
        number_of_beans = action[1]
        heap_size = self.state[heap_number]

        assert heap_size >= number_of_beans, 'Heap is not big enough' + \
                                             "Heap number is: {}. You tried to take {}, but there are only {} beans " \
                                             "in this heap. These are the heaps: {}".format(
                                                 heap_number, number_of_beans, heap_size, self.state)

        self.state[heap_number] -= number_of_beans

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

    def generate_action_space(self):
        # Action space is two numbers
        # First is heap number, Second is number to remove
        self.action_space = spaces.Box(np.array([0, 1]), np.array([self.numberOfHeaps, self.maxHeapSize]))
        self.get_action_size()
        self.generate_output_to_action_map()
        return self.action_space

    def generate_output_to_action_map(self):
        action_map = []
        for heap in range(len(self.heaps)):
            for bean in range(1, self.heaps[heap] + 1):
                action_map.append([heap, bean])
        self.outputToActionMap = action_map
        pass

    def lookup_action(self, output):
        assert output <= len(self.outputToActionMap), 'Outside of range of possible actions.' + \
                                                      'Output is:{}. Should be in range 0-{}'.format(output, len(
                                                          self.outputToActionMap))
        return self.outputToActionMap[output]

    def get_move_list(self):
        return self.outputToActionMap

    def get_possible_moves(self):
        p_moves = []
        for heap in range(len(self.state)):
            for bean in range(1, self.state[heap] + 1):
                p_moves.append([heap, bean])
        return p_moves

    def get_possible_move_indices(self):
        p_moves = [self.outputToActionMap.index(item) for item in self.outputToActionMap if
                   item in self.get_possible_moves()]
        return p_moves

    def get_illegal_move_indices(self):
        imp_moves = [self.outputToActionMap.index(item) for item in self.outputToActionMap if
                     item not in self.get_possible_moves()]
        return imp_moves

    def get_optimal_moves(self):
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
