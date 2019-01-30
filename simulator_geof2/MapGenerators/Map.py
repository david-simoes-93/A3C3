from simulator_geof2.MapGenerators.Obstacle import Obstacle


class Map(object):
    def __init__(self, obstacles, starting_positions, rewards):
        # all maps should be based on 1280x800 screen, as per original Geometry Friends

        # standard map limits
        self.obstacles = [Obstacle([640, 780], 1280, 40), Obstacle([640, 20], 1280, 40),
                          Obstacle([20, 400], 40, 800), Obstacle([1260, 400], 40, 800)]
        self.obstacles.extend(obstacles)

        # each map should only spawn as many agents as starting positions are available
        self.starting_positions = starting_positions

        # each map should have 3 rewards maximum, for consistency
        self.rewards = rewards

    # this function can be overridden for maps that have points of no return
    def is_terminal(self, positions):
        return len(self.rewards) == 0
