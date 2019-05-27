#!/usr/bin/python3
import socket
import subprocess
import os
import gym
import numpy as np
from time import sleep
import random
import signal
import sys
import traceback
from simulator_fcp.Scenario import GameState, KeepAway


def signal_handler(sig, frame):
    print('Terminating...')
    gym_fcp_kill_all()
    sys.exit(0)


def find_process_id_using_port(portnum):
    fp = os.popen("lsof -i :%s" % portnum)
    lines = fp.readlines()
    fp.close()
    pid = None
    if len(lines) >= 2:
        pid = lines[1].split()[1]
    return pid


def gym_fcp_kill_all():
    os.system("killall -9 deepAgent")
    os.system("killall -9 rcssserver3d")


class GymFCPKeepAway(gym.Env):
    def __init__(self, debug=False, serverports=[3100, 3200]):
        self.scenario = KeepAway()
        self.debug = debug

        # kill any existing rcss in this port
        rcss_id = find_process_id_using_port(serverports[0])
        if rcss_id is not None:
            subprocess.Popen(("kill -9 " + rcss_id).split()).wait()

        # info
        self.joints = ["head1", "head2", "lleg1", "rleg1", "lleg2", "rleg2", "lleg3", "rleg3", "lleg4", "rleg4",
                       "lleg5", "rleg5", "lleg6", "rleg6", "larm1", "rarm1", "larm2", "rarm2", "larm3", "rarm3",
                       "larm4", "rarm4", "lleg7", "rleg7"]
        self.number_of_joints = len(self.joints)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("localhost", 0))
        self.server_socket.listen(0)
        self.server_socket.settimeout(10)
        _, self.server_socket_port = self.server_socket.getsockname()
        self.client_socket0 = None
        self.client_socket1 = None
        self.client_socket2 = None
        self.client_socket_oppo = None
        self.cycles_per_second = 50
        self.scenario_time = self.cycles_per_second * 3

        # current working dir
        self.cwd = os.getcwd().split("A3C3")[0] + "A3C3/simulator_fcp/fcp/"
        if self.debug:
            print("CWD: ", self.cwd)

        # start RCSS
        self.rcss_process = None
        self.original_server_ports = serverports
        self.server_port = serverports[0]
        self.server_monitor_port = serverports[1]
        # self.start_rcss()

        self.rewards_sum = 0

        self.agent_process0 = None
        self.agent_process1 = None
        self.agent_process2 = None
        self.agent_process_oppo = None

        # GUI
        self.metadata = {'render.modes': []}

        # Public GYM variables
        self.action_space = self.scenario.action_space
        self.observation_space = self.scenario.observation_space
        self.max_actions = 5

        # A tuple corresponding to the min and max possible rewards
        self.reward_range = [0, 1]

        self.episode_counter = 0
        self.crash_counter = 0

        self.counter = 0

        signal.signal(signal.SIGINT, signal_handler)

    def start_rcss(self):
        self.counter = 0

        if self.debug:
            self.rcss_process = subprocess.Popen(("/usr/local/bin/rcssserver3d --agent-port " + str(self.server_port) +
                                                  " --server-port " + str(self.server_monitor_port)).split(),
                                                 preexec_fn=os.setsid)
        else:
            self.rcss_process = subprocess.Popen(("/usr/local/bin/rcssserver3d --agent-port " + str(self.server_port) +
                                                  " --server-port " + str(self.server_monitor_port)).split(),
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                 preexec_fn=os.setsid)
        sleep(3)
        print("Started RCSS")

    def render(self, mode='human', close=False):
        if close:
            return

        super(GymFCPKeepAway, self).render(mode=mode)
        return

    def refresh_agents(self):
        zero_op = "0".encode("utf-8")

        if self.client_socket0 is not None:
            # print(self.client_socket0.recv(1024).decode("utf-8"))
            self.client_socket0.sendall(zero_op)

        if self.client_socket1 is not None:
            # print(self.client_socket1.recv(1024).decode("utf-8"))
            self.client_socket1.sendall(zero_op)

        if self.client_socket2 is not None:
            # print(self.client_socket2.recv(1024).decode("utf-8"))
            self.client_socket2.sendall(zero_op)

        if self.client_socket_oppo is not None:
            # print(self.client_socket2.recv(1024).decode("utf-8"))
            self.client_socket_oppo.sendall(zero_op)

    def spawn(self, args):
        # self.refresh_agents()
        # agent_process = 0
        print(args)
        # sleep(10)
        # print("waiting...")
        if self.debug:
            agent_process = subprocess.Popen(args.split(),
                                             preexec_fn=os.setsid, cwd=self.cwd)
        else:
            # agent_process = 0
            # sleep(10)
            agent_process = subprocess.Popen(args.split(),
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                             preexec_fn=os.setsid, cwd=self.cwd)

        sleep(3)
        # self.refresh_agents()
        print("waiting for agent to connect...")
        try:
            (client_socket, address) = self.server_socket.accept()
        except:
            print("--error--:", sys.exc_info()[0])
            return None, None

        client_socket.settimeout(20)
        return agent_process, client_socket

    def reset(self):
        # reset every 30 episodes
        self.counter += 1
        if self.counter % 30 == 0:
            self.recover_from_crash()

        self.episode_counter += 1

        # starts/resets agents
        if self.agent_process0 is None or self.agent_process1 is None or self.agent_process2 is None \
                or self.agent_process_oppo is None:

            # restart FCP
            self.start_rcss()
            print("starting agents")
            global_args = "./deepAgent -p " + str(self.server_port) + " " + str(self.server_monitor_port) + \
                          " -dp " + str(self.server_socket_port)
            args0 = global_args + self.scenario.args0
            args1 = global_args + self.scenario.args1
            args2 = global_args + self.scenario.args2
            args_oppo = global_args + self.scenario.args_oppo

            # print("Waiting for FCP Agent to be started")

            # self.debug = True

            if self.agent_process0 is None:
                self.agent_process0, self.client_socket0 = self.spawn(args0)
            # self.debug = False
            if self.agent_process1 is None:
                # print("going for agent1")
                self.agent_process1, self.client_socket1 = self.spawn(args1)

            if self.agent_process2 is None:
                # print("going for agent2")
                self.agent_process2, self.client_socket2 = self.spawn(args2)

            if self.agent_process_oppo is None:
                # print("going for agent2")
                self.agent_process_oppo, self.client_socket_oppo = self.spawn(args_oppo)

            if self.agent_process0 is None or self.agent_process1 is None or \
                            self.agent_process2 is None or self.agent_process_oppo is None:
                print("Agents crashed during spawn!")
                self.recover_from_crash()
                return self.reset()

                # self.debug = False
        else:
            self.reset_agent()

        if self.debug:
            print("Syncing agents")

        self.refresh_agents()
        sleep(1)

        specific_state0, specific_state1, specific_state2, specific_state_oppo, \
        game_state0, game_state1, game_state2 = self.read_state_from_rcss()
        self.state0, self.state1, self.state2 = specific_state0, specific_state1, specific_state2
        self.game_state0, self.game_state1, self.game_state2 = game_state0, game_state1, game_state2

        if specific_state0 is None or specific_state1 is None or specific_state2 is None or specific_state_oppo is None:
            # Agent crashed
            print("Someone crashed!", specific_state0, specific_state1, specific_state2, specific_state_oppo)
            self.recover_from_crash()
            return self.reset()

        self.rewards_sum = 0

        return [specific_state0, specific_state1, specific_state2], \
               {"state_central": self.get_central_state(self.state0, self.state1, self.state2)}

    def recover_from_crash(self):
        self.crash_counter += 1
        print("Recovering from crash ", self.crash_counter, "out of", self.episode_counter, "episodes")
        self.close()  # Close everything
        self.server_port = self.original_server_ports[0] + self.crash_counter % 100
        self.server_monitor_port = self.original_server_ports[1] + self.crash_counter % 100
        # self.start_rcss()  # Start server again
        # return self.reset()  # Reset environment

    def read_message(self):
        # sleep(1)

        buffer0 = ""
        buffer1 = ""
        buffer2 = ""
        buffer_oppo = ""

        try:
            msg_bytes = self.client_socket0.recv(4)
            if len(msg_bytes) == 0:
                raise socket.error("FCP closed conn")
            bytes_to_read = int.from_bytes(msg_bytes, "big")
            # print("reading ",msg_bytes, int.from_bytes(msg_bytes, "big"))
            msg_bytes = self.client_socket0.recv(bytes_to_read)
            while len(msg_bytes) < bytes_to_read:
                print("read only", len(msg_bytes), "reading further", bytes_to_read - len(msg_bytes))
                msg_bytes += self.client_socket0.recv(bytes_to_read - len(msg_bytes))
            buffer0 += msg_bytes.decode("utf-8")
            # print(len(buffer0), buffer0, len(buffer0.split(" ")))
            # print("py 0:", buffer0)
        except socket.error as err:
            print("Socket 0 timeout?")
            print(err)

        try:
            msg_bytes = self.client_socket1.recv(4)
            if len(msg_bytes) == 0:
                raise socket.error("FCP closed conn")
            bytes_to_read = int.from_bytes(msg_bytes, "big")
            # print("reading ", msg_bytes, int.from_bytes(msg_bytes, "big"))
            msg_bytes = self.client_socket1.recv(int.from_bytes(msg_bytes, "big"))
            while len(msg_bytes) < bytes_to_read:
                print("read only", len(msg_bytes), "reading further", bytes_to_read - len(msg_bytes))
                msg_bytes += self.client_socket1.recv(bytes_to_read - len(msg_bytes))
            buffer1 += msg_bytes.decode("utf-8")
            # print(len(buffer1), buffer1, len(buffer1.split(" ")))
            # print("py 1:", buffer1)
        except socket.error as err:
            print("Socket 1 timeout?")
            print(err)

        try:
            msg_bytes = self.client_socket2.recv(4)
            if len(msg_bytes) == 0:
                raise socket.error("FCP closed conn")
            bytes_to_read = int.from_bytes(msg_bytes, "big")
            # print("reading ", msg_bytes, int.from_bytes(msg_bytes, "big"))
            msg_bytes = self.client_socket2.recv(int.from_bytes(msg_bytes, "big"))
            while len(msg_bytes) < bytes_to_read:
                print("read only", len(msg_bytes), "reading further", bytes_to_read - len(msg_bytes))
                msg_bytes += self.client_socket2.recv(bytes_to_read - len(msg_bytes))
            buffer2 += msg_bytes.decode("utf-8")
            # print(len(buffer2), buffer2, len(buffer2.split(" ")))
            # print("py 2:", buffer2)
        except socket.error as err:
            print("Socket 2 timeout?")
            print(err)

        try:
            msg_bytes = self.client_socket_oppo.recv(4)
            if len(msg_bytes) == 0:
                raise socket.error("FCP closed conn")
            bytes_to_read = int.from_bytes(msg_bytes, "big")
            # print("reading ", msg_bytes, int.from_bytes(msg_bytes, "big"))
            msg_bytes = self.client_socket_oppo.recv(int.from_bytes(msg_bytes, "big"))
            while len(msg_bytes) < bytes_to_read:
                # print("read only", len(msg_bytes), "reading further", bytes_to_read - len(msg_bytes))
                msg_bytes += self.client_socket_oppo.recv(bytes_to_read - len(msg_bytes))
            buffer_oppo += msg_bytes.decode("utf-8")
            # print(len(buffer2), buffer2, len(buffer2.split(" ")))
            # print("py 2:", buffer2)
        except socket.error as err:
            print("Socket oppo timeout?")
            print(err)

        if self.debug:
            print("PYTHON READ: " + buffer0 + buffer1 + buffer2)

        return buffer0, buffer1, buffer2, buffer_oppo

    def read_state_from_rcss(self):
        buffer0, buffer1, buffer2, buffer_oppo = self.read_message()

        specific_state0, specific_state1, specific_state2, specific_state_oppo = None, None, None, None
        game_state0, game_state1, game_state2 = None, None, None
        if len(buffer0) != 0:
            state = [float(x) if np.isfinite(float(x)) else 0 for x in buffer0.strip().split(" ")]
            if len(state) != 1:
                my_state0 = state[15:]
                game_state0 = GameState(state[0:15])

                specific_state0 = self.scenario.get_state(my_state0, game_state0)

                self.state0 = specific_state0
                self.game_state0 = game_state0
            else:
                specific_state0 = [0]
        else:
            print("agent0 got empty message")

        if len(buffer1) != 0:
            state = [float(x) if np.isfinite(float(x)) else 0 for x in buffer1.strip().split(" ")]
            if len(state) != 1:
                my_state1 = state[15:]
                game_state1 = GameState(state[0:15])

                specific_state1 = self.scenario.get_state(my_state1, game_state1)

                self.state1 = specific_state1
                self.game_state1 = game_state1
            else:
                specific_state1 = [0]
        else:
            print("agent1 got empty message")

        if len(buffer2) != 0:
            state = [float(x) if np.isfinite(float(x)) else 0 for x in buffer2.strip().split(" ")]
            if len(state) != 1:
                my_state2 = state[15:]
                game_state2 = GameState(state[0:15])

                specific_state2 = self.scenario.get_state(my_state2, game_state2)

                self.state2 = specific_state2
                self.game_state2 = game_state2
            else:
                specific_state2 = [0]
        else:
            print("agent2 got empty message")

        if len(buffer_oppo) != 0:
            state = [float(x) if np.isfinite(float(x)) else 0 for x in buffer2.strip().split(" ")]
            if len(state) != 1:
                my_state_oppo = state[15:]
                game_state_oppo = GameState(state[0:15])

                specific_state_oppo = self.scenario.get_state(my_state_oppo, game_state_oppo)

                self.state_oppo = specific_state_oppo
                self.game_state_oppo = game_state_oppo
            else:
                specific_state_oppo = [0]
        else:
            print("agent oppo got empty message")

        if self.debug:
            print("AGENT STATE:", specific_state0, specific_state1, specific_state2, specific_state_oppo)

        return specific_state0, specific_state1, specific_state2, specific_state_oppo, \
               game_state0, game_state1, game_state2

    def debugsend(self, st):
        self.client_socket.sendall(st.encode("utf-8"))

    def get_central_state(self, state0, state1, state2):
        game_state_updated = [self.game_state0.my_pos_x / 10, self.game_state0.my_pos_y / 10,
                              self.game_state1.my_pos_x / 10, self.game_state1.my_pos_y / 10,
                              self.game_state2.my_pos_x / 10, self.game_state2.my_pos_y / 10]
        if state0 is not None and len(state0) != 1:
            game_state_updated.extend(
                [self.game_state0.ball_x / 10, self.game_state0.ball_y / 10])
        elif state1 is not None and len(state1) != 1:
            game_state_updated.extend(
                [self.game_state1.ball_x / 10, self.game_state1.ball_y / 10])
        elif state2 is not None and len(state2) != 1:
            game_state_updated.extend(
                [self.game_state2.ball_x / 10, self.game_state2.ball_y / 10])
        else:
            print("Incomplete game state:", state0, state1, state2)
            traceback.print_stack()
            game_state_updated.extend([0, 0])
        game_state_updated.extend([self.game_state_oppo.my_pos_x / 10, self.game_state_oppo.my_pos_y / 10])

        if np.isnan(game_state_updated).any():
            print("Found NaN, game_state_updated:")
            print(game_state_updated, state0, state1, state2)

        low = [-1.5, -1, -1.5, -1, -1.5, -1, -1.5, -1, -1.5, -1]
        high = [1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1]
        for i in range(len(game_state_updated)):
            if game_state_updated[i] < low[i]:
                game_state_updated[i] = low[i]
            elif game_state_updated[i] > high[i]:
                game_state_updated[i] = high[i]

        return game_state_updated

    def check_crash(self):
        # Agent crashed
        self.recover_from_crash()
        # self.crash_counter += 1
        # print("Crash ", self.crash_counter, "out of", self.episode_counter, "episodes")
        # self.kill_agents()
        return [np.zeros(len(self.observation_space.low)),
                np.zeros(len(self.observation_space.low)),
                np.zeros(len(self.observation_space.low))], -1, True, \
               {"state_central": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

    def step(self, actions):
        if actions[0] is not None:
            self.client_socket0.sendall(str(actions[0]).encode("utf-8"))
        if actions[1] is not None:
            self.client_socket1.sendall(str(actions[1]).encode("utf-8"))
        if actions[2] is not None:
            self.client_socket2.sendall(str(actions[2]).encode("utf-8"))

        self.client_socket_oppo.sendall("5".encode("utf-8"))

        if self.debug:
            print("Python sent", actions)

        state0, state1, state2, state_oppo, game_state0, game_state1, game_state2 = self.read_state_from_rcss()
        if state0 is None or state1 is None or state2 is None or state_oppo is None:
            return self.check_crash()

        while len(state0) == 1 and len(state1) == 1 and len(state2) == 1:
            # print("reading")
            self.refresh_agents()
            state0, state1, state2, state_oppo, game_state0, game_state1, game_state2 = self.read_state_from_rcss()
            # print("read", state0, state1, state2, game_state)

            if state0 is None or state1 is None or state2 is None or state_oppo is None:
                return self.check_crash()

        terminal, reward = self.scenario.get_terminal_reward([self.state0, self.state1, self.state2, self.state_oppo],
                                                             [self.game_state0, self.game_state1, self.game_state2])

        return [state0, state1, state2], reward, terminal, \
               {"state_central": self.get_central_state(state0, state1, state2)}

    def reset_agent(self):
        if self.client_socket0 is not None:
            self.client_socket0.sendall("reset".encode("utf-8"))
            # print("Reset agent")
        if self.client_socket1 is not None:
            self.client_socket1.sendall("reset".encode("utf-8"))
            # print("Reset agent")
        if self.client_socket2 is not None:
            self.client_socket2.sendall("reset".encode("utf-8"))
            # print("Reset agent")
        if self.client_socket_oppo is not None:
            self.client_socket_oppo.sendall("reset".encode("utf-8"))
            # print("Reset agent")

    def kill_agents(self):
        self.kill_agent(self.client_socket0, self.agent_process0)
        self.client_socket0 = None
        self.agent_process0 = None

        self.kill_agent(self.client_socket1, self.agent_process1)
        self.client_socket1 = None
        self.agent_process1 = None

        self.kill_agent(self.client_socket2, self.agent_process2)
        self.client_socket2 = None
        self.agent_process2 = None

        self.kill_agent(self.client_socket_oppo, self.agent_process_oppo)
        self.client_socket_oppo = None
        self.agent_process_oppo = None

    def kill_agent(self, client_socket, agent_process):
        if client_socket is not None:
            try:
                client_socket.sendall("kill".encode("utf-8"))
            except BrokenPipeError:
                print("Socket was already closed")
            except Exception as e:
                print("Error killing agent:", e)
            client_socket.close()
            sleep(2)

        if agent_process is not None:
            deep_pid = agent_process.pid
            agent_process.kill()
            # self.agent_process = None
            sleep(1)
            subprocess.Popen(("kill -9 " + str(deep_pid)).split()).wait()

        print("Killed agent")

    def close(self):
        self.render(close=True)

        # close FCP
        self.kill_agents()

        if self.rcss_process is not None:
            self.rcss_process.kill()
            self.rcss_process = None
            sleep(1)

        rcss_id = find_process_id_using_port(self.server_port)
        if rcss_id is not None:
            subprocess.Popen(("kill -9 " + rcss_id).split()).wait()
        print("Killed RCSS")

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(sys.maxsize)
        random.seed(seed)
        return [seed]
