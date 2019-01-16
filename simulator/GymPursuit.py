#!/usr/bin/python3

import gym
from gym.spaces import *
import numpy as np
import random
import sys
from time import sleep
import pygame


class GymPursuit(gym.Env):
    def __init__(self, number_of_agents=4, obs_radius=5, number_of_prey=4, map_size=17):
        # super?
        self.max_actions = 5
        self.number_of_agents = number_of_agents
        self.number_of_prey = number_of_prey
        self.map_size = map_size
        self.obs_size = min(obs_radius, map_size)

        # Global info
        self.rewards = []
        self.terminal = False

        self.pos = []
        self.prey_pos = []
        self.prev_prey_pos = []
        self.closest_preds_to_prey = []
        for i in range(self.number_of_agents):
            self.pos += [[]]
        for i in range(self.number_of_prey):
            self.prey_pos += [[]]

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

        self.agent_observation_space = [2 + obs_radius * obs_radius * 2]
        self.central_observation_space = [(number_of_prey + number_of_agents) * 2]
        self.agent_action_space = self.max_actions

        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [0, self.number_of_prey]

        self.timer = 0
        self.agent_colors = [(random.randint(128, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                             range(number_of_agents)]
        # self.max_timer = 1

    def render(self, mode='human', close=False):
        cell_width = 50
        cell_width_half = int(cell_width / 2)
        cell_width_quarter = int(cell_width / 8)
        obs_render_octo = [[0, 0], [-cell_width * self.map_size, 0], [cell_width * self.map_size, 0],
                           [0, -cell_width * self.map_size], [0, cell_width * self.map_size],
                           [cell_width * self.map_size, cell_width * self.map_size],
                           [-cell_width * self.map_size, cell_width * self.map_size],
                           [cell_width * self.map_size, -cell_width * self.map_size],
                           [-cell_width * self.map_size, -cell_width * self.map_size]]

        if close:
            pygame.quit()
            self.screen = None
            return

        if mode is not 'human':
            super(GymPursuit, self).render(mode=mode)
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode([cell_width * self.map_size, cell_width * self.map_size])
            pygame.display.set_caption("Blind GroupUp")

        self.screen.fill((255, 255, 255))

        # Draw agents
        obs_half_cells = int((self.obs_size - 1) / 2)
        for agent_index, agent in enumerate(self.pos):
            pygame.draw.rect(self.screen, self.agent_colors[agent_index],
                             [agent[0] * cell_width, agent[1] * cell_width, cell_width, cell_width])
            for octo in obs_render_octo:
                pygame.draw.rect(self.screen, self.agent_colors[agent_index],
                                 [(agent[0] - obs_half_cells) * cell_width + octo[0],
                                  (agent[1] - obs_half_cells) * cell_width + octo[1],
                                  cell_width * self.obs_size, cell_width * self.obs_size], 1)

        # draw target
        for agent_index, agent in enumerate(self.prey_pos):
            pygame.draw.circle(self.screen, (0, 255, 0),
                               [agent[0] * cell_width + cell_width_half, agent[1] * cell_width + cell_width_half],
                               cell_width_half)

        for prey_index, agent_index in enumerate(self.closest_preds_to_prey):
            pygame.draw.circle(self.screen, (0, 0, 0),
                               [self.pos[agent_index][0] * cell_width + cell_width_half,
                                self.pos[agent_index][1] * cell_width + cell_width_half],
                               cell_width_quarter)

        pygame.display.flip()

    def reset(self):
        # self.max_timer += 1
        # print("max timer pursuit",self.max_timer)
        self.timer = 0

        self.pos = []
        self.prey_pos = []
        self.closest_preds_to_prey = []
        for i in range(self.number_of_agents):
            self.pos += [[]]
        for i in range(self.number_of_prey):
            self.prey_pos += [[]]

        for i in range(self.number_of_agents):
            new_test_pos = []
            while new_test_pos in self.pos:
                new_test_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.pos[i] = new_test_pos

        for i in range(self.number_of_prey):
            new_test_pos = []
            while new_test_pos in self.prey_pos or new_test_pos in self.pos:
                new_test_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
            self.prey_pos[i] = new_test_pos

        return self.get_state(), {"state_central": self.get_state_central(), "time_chasing": 0}

    def step(self, action):
        reward = -0.1
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

        for pos in self.pos:
            # find prey caught
            if pos in self.prey_pos:
                self.prey_pos.remove(pos)
                reward += 1

            # find collisions
            if self.pos.count(pos) > 1:
                reward -= 0.5
                new_test_pos = pos
                while new_test_pos in self.pos or new_test_pos in self.prey_pos:
                    new_test_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                pos[0] = new_test_pos[0]
                pos[1] = new_test_pos[1]

        terminal = len(self.prey_pos) == 0

        # self._render()
        # sleep(0.5)
        self.step_prey()

        # find prey caught
        for pos in self.pos:
            if pos in self.prey_pos:
                # index_of_prey = self.prey_pos.index(pos)
                # self.closest_preds_to_prey.pop(index_of_prey)
                self.prey_pos.remove(pos)

                reward += 1

        # find collisions
        for pos in self.prey_pos:
            if self.prey_pos.count(pos) > 1:
                new_test_pos = pos
                while new_test_pos in self.pos or new_test_pos in self.prey_pos:
                    new_test_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                pos[0] = new_test_pos[0]
                pos[1] = new_test_pos[1]


        # if self.timer >= self.max_timer:
        #    terminal = True

        return self.get_state(), reward, terminal, {"state_central": self.get_state_central(),
                                                    "time_chasing": self.timer}

    def step_prey(self):
        self.closest_preds_to_prey = []

        # self.prev_prey_pos = []
        # for prey_pos in self.prey_pos:
        #    self.prev_prey_pos.append(list(prey_pos))

        for prey_pos in self.prey_pos:
            pos0 = list(self.pos[0])
            self.centralize(prey_pos, pos0)

            # find closest predator
            closest_dist, closest_pred, closest_index = dist(prey_pos, pos0), pos0, 0
            for index, pred in zip(range(1, len(self.pos)), self.pos[1:]):
                this_pred = list(pred)
                self.centralize(prey_pos, this_pred)
                this_dist = dist(prey_pos, this_pred)
                if this_dist < closest_dist:
                    closest_dist = this_dist
                    closest_pred = this_pred
                    closest_index = index
            self.closest_preds_to_prey.append(closest_index)

            # move away
            delta_x = prey_pos[0] - closest_pred[0]
            delta_y = prey_pos[1] - closest_pred[1]

            if np.abs(delta_x) > np.abs(delta_y):
                if delta_y > 0:
                    prey_pos[1] = (prey_pos[1] + 1) % self.map_size
                elif delta_y < 0:
                    prey_pos[1] = (prey_pos[1] - 1) % self.map_size
                else:
                    prey_pos[1] = (prey_pos[1] + (1 if random.uniform(0, 1) > 0.5 else -1)) % self.map_size
            elif np.abs(delta_x) < np.abs(delta_y):
                if delta_x > 0:
                    prey_pos[0] = (prey_pos[0] + 1) % self.map_size
                elif delta_x < 0:
                    prey_pos[0] = (prey_pos[0] - 1) % self.map_size
                else:
                    prey_pos[0] = (prey_pos[0] + (1 if random.uniform(0, 1) > 0.5 else -1)) % self.map_size
            else:
                if random.uniform(0, 1) > 0.5:
                    prey_pos[1] = (prey_pos[1] + (1 if delta_y > 0 else -1)) % self.map_size
                else:
                    prey_pos[0] = (prey_pos[0] + (1 if delta_x > 0 else -1)) % self.map_size

    def get_state_central(self):
        map = []
        for pos in self.pos:
            map.extend(pos)
        for pos in self.prey_pos:
            map.extend(pos)
        for i in range(len(self.prey_pos), self.number_of_prey):
            map.extend([-1, -1])
        return [map for _ in range(self.number_of_agents)]

    # computes the agent's observations
    def get_state(self):
        obs = []
        prey_within_view = False
        center_position_of_local_view = int((self.obs_size - 1) / 2)
        center_position_of_global_view = int((self.map_size - 1) / 2)
        for index, pos in enumerate(self.pos):
            my_obs = np.zeros([self.obs_size, self.obs_size, 2])
            translation = [-pos[0] + center_position_of_local_view, -pos[1] + center_position_of_local_view]

            for other_index, other_pos in enumerate(self.pos):
                other_pos = list(other_pos)
                # adjust positions so they are centered around agent
                self.centralize(pos, other_pos)

                # print(other_index, other_pos, pos, translation)
                relX = other_pos[0] + translation[0]
                relY = other_pos[1] + translation[1]
                if 0 <= relX < self.obs_size and 0 <= relY < self.obs_size:
                    my_obs[relX][relY][0] = 1

            for other_index, other_pos in enumerate(self.prey_pos):
                other_pos = list(other_pos)

                # adjust positions so they are centered around agent
                self.centralize(pos, other_pos)

                # print(other_index, other_pos, pos, translation)
                relX = other_pos[0] + translation[0]
                relY = other_pos[1] + translation[1]
                if 0 <= relX < self.obs_size and 0 <= relY < self.obs_size:
                    my_obs[relX][relY][1] = 1
                    prey_within_view = True
            # for i in range(5):
            #    print(int(my_obs[i][0][0]),int(my_obs[i][1][0]),int(my_obs[i][2][0]),int(my_obs[i][3][0]),int(my_obs[i][4][0]))
            # print()
            obs.append(np.append(my_obs.flatten(),
                                 [pos[0] - center_position_of_global_view, pos[1] - center_position_of_global_view]))
        if prey_within_view and len(self.prey_pos)<=2:
            self.timer += 1

        return obs

    def close(self):
        return

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]

    def centralize(self, pos, other_pos):
        if other_pos[0] < pos[0] - self.map_size / 2:
            other_pos[0] += self.map_size
        if other_pos[0] > pos[0] + self.map_size / 2:
            other_pos[0] -= self.map_size
        if other_pos[1] < pos[1] - self.map_size / 2:
            other_pos[1] += self.map_size
        if other_pos[1] > pos[1] + self.map_size / 2:
            other_pos[1] -= self.map_size


# manhattan distance
def dist(a, b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
