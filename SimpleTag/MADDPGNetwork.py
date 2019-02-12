import socket
import pickle
import time


# MADDPG Network
class MADDPG_Network:
    def __init__(self, ip="localhost", port=10005):
        self.ip = ip
        self.port = port

        self.socket = None
        self.connect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        self.socket.settimeout(20)
        # time.sleep(20)

    def send_obs(self, obs_n):
        data = pickle.dumps(obs_n)

        done = False
        while not done:
            try:
                # print("sending", len(data))
                sendInt(self.socket, len(data))
                # print("sending data")
                sendPickle(self.socket, data)
                # print("reading")
                done = True
            except:
                print("-Socket down")
                self.connect()

    def get_action(self):
        try:
            # print("getting")
            mess_size = readInt(self.socket)
            # print("reading", mess_size)
            action_n = readPickle(self.socket, mess_size)
            # print("got", action_n)
        except:
            print("+Socket down")
            self.connect()
            return [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]

        return action_n


def readInt(sock):
    chunks = b""
    while ";" not in chunks.decode():
        chunk = sock.recv(8)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunks += chunk

    return int(chunks.decode().split(";")[0])


def readPickle(sock, bytes):
    chunks = b""
    bytes_recd = 0
    while bytes_recd < bytes:
        chunk = sock.recv(min(bytes - bytes_recd, 2048))
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunks += chunk
        bytes_recd = bytes_recd + len(chunk)

    return pickle.loads(chunks)


def sendInt(sock, value):
    data = str(value) + ";"
    while len(data) % 8 != 0:
        data += " "
    data = data.encode()

    totalsent = 0
    while totalsent < len(data):
        sent = sock.send(data[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent


def sendPickle(sock, data):
    totalsent = 0
    while totalsent < len(data):
        sent = sock.send(data[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent
