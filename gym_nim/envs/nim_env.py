import gym
from gym import error, spaces, utils
from gym.utils import seeding


class NimEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print("Initialised")
        return

    def step(self, action):
        print("Stepped")
        return

    def reset(self):
        print("Reset")
        return

    def render(self, mode='human', close=False):
        print("Rendered")
        return
