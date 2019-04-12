from gym import spaces
import numpy as np


class GameState():
    def __init__(self, state):
        self.rel_ball_x, self.rel_ball_y = state[0], state[1]
        self.imu_roll, self.imu_pitch, self.imu_yaw = state[2], state[3], state[4]
        self.gyro_x, self.gyro_y, self.gyro_z = state[5], state[6], state[7],
        self.my_ori = state[8]
        self.my_pos_x, self.my_pos_y, self.my_pos_z = state[9], state[10], state[11]
        self.ball_x, self.ball_y = state[12], state[13]
        self.game_time = state[14]


class Scenario():
    def __init__(self):
        self.name = ""
        self.action_space = None
        self.observation_space = None
        self.args = ""

    # returns relevant state for the algorithm
    def get_state(self, joints, prev_actions, game_state):
        return []

    # returns whether episode is terminal, and the reward
    def get_terminal_reward(self, state, game_state):
        return True, 0


class KeepAway(Scenario):
    def __init__(self):
        self.name = "keepaway"

        self.action_space = spaces.Tuple((spaces.Discrete(5), spaces.Discrete(5), spaces.Discrete(5)))
        self.observation_space = spaces.Box(
            low=np.array([-90, -90, -25, -45, -25, -25, -130, -130, -45, -45, -45, -25,  # legs
                          -120, -120, -1, -95, -120, -120, -90, -1,  # arms
                          -360, -360, -360,  # gyro
                          -180, -15, -10, 0]),  # orient, x, y, z
            high=np.array([1, 1, 45, 25, 100, 100, 1, 1, 75, 75, 25, 45,  # legs
                           120, 120, 95, 1, 120, 120, 1, 90,  # arms
                           360, 360, 360,  # gyro
                           180, 15, 10, 1]), dtype=np.float32)
        self.args0 = " -ds keepaway -u 4 -dbeam 0 -9 0 -r 4 -dball 0 -8.9 0"
        self.args1 = " -ds keepaway -u 3 -dbeam -9 9 0 -r 4 -dball 0 -8.9 0"
        self.args2 = " -ds keepaway -u 2 -dbeam 9 9 0 -r 4 -dball 0 -8.9 0"  # doesnt beam there
        self.last_next_to_ball = 2
        self.scenario_time = 10

    def get_state(self, joints, prev_actions, game_state):
        prevPlayerPos = joints[0:2]
        nextPlayerPos = joints[2:4]

        ballPosAfterStopping = joints[4:6]
        myDistToBall = joints[6]
        prevPlayerDistToBall = joints[7]
        nextPlayerDistToBall = joints[8]

        return [game_state.my_pos_x, game_state.my_pos_y] + prevPlayerPos + nextPlayerPos + \
               [game_state.rel_ball_x, game_state.rel_ball_y] + ballPosAfterStopping + \
               [game_state.my_ori, myDistToBall, prevPlayerDistToBall, nextPlayerDistToBall]

    def get_terminal_reward(self, joints, game_states):
        reward = 0
        if self.last_next_to_ball == 0 and joints[0][6] > 2 and joints[1][6] < 1:
            reward = 1
            self.last_next_to_ball = 1
        elif self.last_next_to_ball == 0 and joints[0][6] > 2 and joints[2][6] < 1:
            reward = 1
            self.last_next_to_ball = 2
        elif self.last_next_to_ball == 1 and joints[1][6] > 2 and joints[0][6] < 1:
            reward = 1
            self.last_next_to_ball = 0
        elif self.last_next_to_ball == 1 and joints[1][6] > 2 and joints[2][6] < 1:
            reward = 1
            self.last_next_to_ball = 2
        elif self.last_next_to_ball == 2 and joints[2][6] > 2 and joints[0][6] < 1:
            reward = 1
            self.last_next_to_ball = 0
        elif self.last_next_to_ball == 2 and joints[2][6] > 2 and joints[1][6] < 1:
            reward = 1
            self.last_next_to_ball = 1

        terminal = game_states[0].game_time > 100 or game_states[1].game_time > 100 or game_states[2].game_time > 100

        return terminal, reward
