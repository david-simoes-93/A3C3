#!/usr/bin/python3
import random

import gym
import pygame
import sys
from pygame.locals import *


class GeometryFriends2(gym.Env):
    # agents - List of Agent objects
    # maps - List of MapGenerator objects
    # agent_collision - Whether agents are meant to collide with each other
    # screen_res - Resolution for rendering, use None and don't call render() to disable it
    # graphical_state - True if agents read a global pixel observation of the environment, instead of local continuous vectors
    def __init__(self, agents, maps, agent_collision=False, screen_res=[640, 400], graphical_state=None,
                 graphical_state_res=[80, 50], repeated_actions=1):
        self.agents = agents
        self.maps = maps
        self.agent_collision = agent_collision and len(agents) > 1
        self.screen_res = screen_res
        self.graphical_state = graphical_state
        self.graphical_state_res = graphical_state_res

        self.action_space = gym.spaces.Tuple(tuple([agent.action_space for agent in self.agents]))

        # Global info
        self.terminal = False
        self.repeated_actions = repeated_actions

        # GUI
        self.metadata = {'render.modes': ['human']}
        self.screen, self.gui_window, self.screen_resized = None, None, None
        if graphical_state:
            pygame.init()
            self.screen = pygame.surface.Surface((1280, 800))  # original GF size
            self.screen_resized = pygame.surface.Surface(screen_res)  # original GF size
            pygame.display.set_caption("GeoFriends2")

    def prepare_frame(self):
        self.screen.fill((0, 0, 255))

        # Draw obstacles
        for obs in self.map.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0),
                             [obs.left_x, obs.top_y, obs.right_x - obs.left_x, obs.bot_y - obs.top_y])

        # Draw agents
        for agent in self.agents:
            agent.render(self.screen)

        # Draw rewards
        for reward in self.map.rewards:
            pygame.draw.circle(self.screen, (255, 0, 255), [int(reward[0]), int(reward[1])], 25)

    def render(self, mode='human', close=False):
        if close:
            pygame.quit()
            self.screen = None
            return

        if mode is not 'human':
            super(GeometryFriends2, self).render(mode=mode)
            return

        if self.gui_window is None:
            if not self.graphical_state:
                pygame.init()
                self.screen = pygame.surface.Surface((1280, 800))  # original GF size
                pygame.display.set_caption("GeoFriends2")

            self.gui_window = pygame.display.set_mode(self.screen_res, HWSURFACE | DOUBLEBUF | RESIZABLE)

        if not self.graphical_state:
            self.prepare_frame()

        self.gui_window.blit(pygame.transform.scale(self.screen, self.screen_res), (0, 0))
        pygame.display.flip()

    def reset(self):
        self.terminal = False

        self.map = random.choice(self.maps).generate()

        for i, agent in enumerate(self.agents):
            agent.pos = list(self.map.starting_positions[i])
            agent.reset(self.map.obstacles)

        if self.graphical_state:
            self.prepare_frame()

        return self.compute_observations(), {"obstacles": self.map.obstacles}

    def step_single(self, action):
        reward = 0

        for i, agent in enumerate(self.agents):
            agent.step(action[i])

        if self.agent_collision:
            for i, agent in enumerate(self.agents):
                agent.clear_out_of_obstacles(self.get_other_agent_obstacles(i))

        agents_moved = [None] * len(self.agents)
        for i, agent in enumerate(self.agents):
            agents_moved[i] = agent.clear_out_of_obstacles(self.map.obstacles)

        if self.agent_collision:
            for i, agent in enumerate(self.agents):
                agent.clear_out_of_obstacles(self.get_other_agent_obstacles(i), forbidden_moves=agents_moved[i])

        for i, agent in enumerate(self.agents):
            intersected_rewards = agent.check_rewards(self.map.rewards)
            for intersected_reward in intersected_rewards:
                self.map.rewards.remove(intersected_reward)
                reward += 1

        return reward

    def step(self, action):
        reward = self.step_single(action)

        if self.repeated_actions>1:
            action = [self.agents[i].repeated_movement_indexes[action[i]] for i in range(len(self.agents))]
        for i in range(1, self.repeated_actions):
            reward += self.step_single(action)

        if self.graphical_state:
            self.prepare_frame()

        return self.compute_observations(), \
               reward, \
               self.map.is_terminal([agent.pos for agent in self.agents]), \
               {"obstacles": self.map.obstacles}    # alternar entre states

    def compute_observations(self):
        if self.graphical_state:
            self.screen_resized = pygame.transform.scale(self.screen, self.graphical_state_res)
            return pygame.surfarray.array3d(self.screen_resized)
        else:
            observations = []
            for index in range(len(self.agents)):
                curr_obs = self.agents[index].get_state()
                for agent in self.agents[0:index] + self.agents[index + 1:]:
                    curr_obs += agent.get_external_state()
                for reward in self.map.rewards:
                    curr_obs += reward
                observations.append(curr_obs)
        return observations

    def close(self):
        self.render(close=True)
        return

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]

    def get_other_agent_obstacles(self, current_agent_index):
        all_other_obstacles = []
        for j, other_agent in enumerate(self.agents):
            if current_agent_index == j:
                continue
            all_other_obstacles.extend(other_agent.get_obstacle_body())
        return all_other_obstacles
