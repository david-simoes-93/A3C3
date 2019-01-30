from abc import ABC, abstractmethod


class Agent(ABC):
    # Executes an action on the environment
    @abstractmethod
    def step(self, action):
        return

    # Moves the agent outside of obstacles.
    # If agent collision is enabled, this is first called against other agents, then against actual obstacles,
    #    then agents again with forbidden moves. Otherwise, its only called against actual obstacles
    # forbidden_moves=[Right, Left, Down, Up] is an array of directions the agent can no longer move to
    @abstractmethod
    def clear_out_of_obstacles(self, obstacles, forbidden_moves=[False, False, False, False]):
        return False

    # Returns a list of rewards the agent is intersecting with
    @abstractmethod
    def check_rewards(self, rewards):
        return []

    # Returns the agent's state (like position or other known variables)
    # Obstacles and rewards are added later by the environment
    @abstractmethod
    def get_state(self):
        return None

    # Returns the agent's state (like position) relevant for other agents
    @abstractmethod
    def get_external_state(self):
        return None

    # Renders the agent on a PyGame screen
    @abstractmethod
    def render(self, screen):
        return

    # Reset agents variables
    @abstractmethod
    def reset(self, obstacles):
        return

    # Returns the agent's shape in terms of obstacles that other agents collide with
    @abstractmethod
    def get_obstacle_body(self):
        return []


