#!/usr/bin/python3

import gym
from gym.spaces import *
import numpy as np
import random
import sys
import pygame


class GymBGU(gym.Env):
    def __init__(self, number_of_agents=11, map_size=5):
        # super?
        self.max_actions = 5
        self.number_of_agents = number_of_agents
        self.map_size = map_size

        # Global info
        self.rewards = []
        self.terminal = False
        self.max_step_limit = map_size * 3
        self.max_step_limit_since_target = map_size * 1.5

        self.pos = []
        self.target_goal = None
        for i in range(self.number_of_agents):
            self.pos += [[]]
        self.timer = 0

        # GUI
        self.metadata = {'render.modes': ['human']}
        self.screen = None

        # Public GYM variables
        # The Space object corresponding to valid actions
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(self.max_actions) for x in range(self.number_of_agents)])
        # (gym.spaces.Discrete(self.max_actions), gym.spaces.Discrete(self.max_actions)))
        # The Space object corresponding to valid observations
        self.observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(self.number_of_agents, 2))

        self.agent_observation_space = [3]
        self.central_observation_space = [3 * number_of_agents]
        self.agent_action_space = self.max_actions

        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [0, self.number_of_agents]

    def render(self, mode='human', close=False):
        cell_width = 50
        cell_width_half = int(cell_width / 2)
        if close:
            pygame.quit()
            self.screen = None
            return

        if mode is not 'human':
            super(GymBGU, self).render(mode=mode)
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode([cell_width * self.map_size, cell_width * self.map_size])
            pygame.display.set_caption("Blind GroupUp")

        self.screen.fill((255, 255, 255))

        # Draw agents
        for agent_index, agent in enumerate(self.pos):
            pygame.draw.circle(self.screen, (0,0,0),
                               [agent[0] * cell_width + cell_width_half, agent[1] * cell_width + cell_width_half],
                               cell_width_half)

        # draw target
        pygame.draw.circle(self.screen, (255, 0, 0),
                           [self.target_goal[0] * cell_width + cell_width_half,
                            self.target_goal[1] * cell_width + cell_width_half],
                           int(cell_width_half / 2))

        pygame.display.flip()

    def reset(self):
        self.target_goal = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        for i in range(self.number_of_agents):
            self.pos[i] = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        #self.pos = [[4, 4], [3, 1], [4, 1], [4, 2]]
        #self.target_goal = [4,4]
        self.timer = 0
        self.found_target = False

        return self.get_state(), {"state_central": self.get_state_central()}

    def step(self, action):
        reward = []
        for i, action in enumerate(action):
            if action == 0:
                mov = [0, 1]
            elif action == 1:
                mov = [0, -1]
            elif action == 2:
                mov = [1, 0]
            elif action == 3:
                mov = [-1, 0]
            else:
                mov = [0, 0]
            self.pos[i][0] = (self.pos[i][0] + mov[0]) % self.map_size
            self.pos[i][1] = (self.pos[i][1] + mov[1]) % self.map_size

        agents_on_target = self.pos.count(self.target_goal)
        for i in range(self.number_of_agents):
            reward.append(agents_on_target if self.target_goal == self.pos[i] else 0)

        if not self.found_target and agents_on_target != 0:
            self.found_target = True
            self.found_target_at = self.timer

        # reward = np.mean(reward)
        if (not self.found_target and self.timer >= self.max_step_limit) or \
                (self.found_target and self.timer >= self.found_target_at+self.max_step_limit_since_target):
            # max([self.pos.count(i) for i in self.pos])
            terminal = True
            # print(reward)
        else:
            # reward = 0
            terminal = False

        self.timer += 1

        return self.get_state(), reward, terminal, {"state_central": self.get_state_central()}

    def get_state_central(self):
        #print([1 if self.target_goal == pos else 0 for pos in self.pos] + [item for sublist in self.pos for item in
        #                                                                    sublist])
        return [1 if self.target_goal == pos else 0 for pos in self.pos] + [item for sublist in self.pos for item in
                                                                            sublist]

    # computes the circle's observations
    def get_state(self):
        # print(self.target_goal, self.pos)
        obs = []
        for index, pos in enumerate(self.pos):
            obs.append([1 if self.target_goal == pos else 0] + pos)
        return obs

    def close(self):
        return

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]
