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
            low=np.array([-1.5, -1, -1.5, -1, -1.5, -1,   -1.5, -1, -1.5, -1, -1, -1,   0, 0, 0]),  # orient, x, y, z
            high=np.array([1.5, 1, 1.5, 1, 1.5, 1,  1.5, 1, 1.5, 1, 1, 1,   1.5, 1.5, 1.5]), dtype=np.float32)
        self.args0 = " -ds keepaway -u 4 -dbeam 0 -9 0 -r 4 -dball 0 -8.9 0"
        self.args1 = " -ds keepaway -u 3 -dbeam -9 9 0 -r 4 -dball 0 -8.9 0"
        self.args2 = " -ds keepaway -u 2 -dbeam 9 9 0 -r 4 -dball 0 -8.9 0"  # doesnt beam there
        self.last_next_to_ball = 2
        self.scenario_time = 10

    def get_state(self, joints, prev_actions, game_state):
        prevPlayerPos = [joints[0]/10, joints[1]/10]
        nextPlayerPos = [joints[2]/10, joints[3]/10]

        ballPosAfterStopping = [joints[4]/10, joints[5]/10]
        #print("get state",joints[6:9])
        myDistToBall = joints[6]/10
        prevPlayerDistToBall = joints[7]/10
        nextPlayerDistToBall = joints[8]/10

        radian_ori = np.math.radians(game_state.my_ori)

        # TODO polar coords?
        state = [game_state.my_pos_x/10, game_state.my_pos_y/10] + prevPlayerPos + nextPlayerPos + \
               [game_state.rel_ball_x/10, game_state.rel_ball_y/10] + ballPosAfterStopping + \
               [np.math.cos(radian_ori), np.math.sin(radian_ori),
                myDistToBall, prevPlayerDistToBall, nextPlayerDistToBall]

        # truncate
        for i in range(len(state)):
            if state[i]<self.observation_space.low[i]:
                state[i] = self.observation_space.low[i]
            elif state[i]>self.observation_space.high[i]:
                state[i] = self.observation_space.high[i]

        if np.isnan(state).any():
            print("Found NaN! State:", state)
            state = [x if np.isfinite(x) else 0 for x in state]

        return state

    def get_terminal_reward(self, joints, game_states):
        dist0 = joints[0][12]*10
        dist1 = joints[1][12]*10
        dist2 = joints[2][12]*10

        reward = 0
        if self.last_next_to_ball == 0 and dist0 > 2:
            if dist1 < 1:
                reward = 1
                self.last_next_to_ball = 1
            elif dist2 < 1:
                reward = 1
                self.last_next_to_ball = 2
        elif self.last_next_to_ball == 1 and dist1 > 2:
            if dist0 < 1:
                reward = 1
                self.last_next_to_ball = 0
            elif dist2 < 1:
                reward = 1
                self.last_next_to_ball = 2
        elif self.last_next_to_ball == 2 and dist2 > 2:
            if dist0 < 1:
                reward = 1
                self.last_next_to_ball = 0
            elif dist1 < 1:
                reward = 1
                self.last_next_to_ball = 1

        """if reward == 1:
            print("reward! ",self.last_next_to_ball)
        else:
            print("prev:",self.last_next_to_ball,"dists:",dist0,dist1,dist2)"""

        terminal = game_states[0].game_time > 100 or game_states[1].game_time > 100 or game_states[2].game_time > 100

        return terminal, reward
