class Obstacle(object):
    # Define an obstacle by defining coordinates of top-left corner and bottom-right corners
    #   Obstacle( [0, 0], [10, 20])
    def __init__(self, top_left, bottom_right):
        self.top_y = top_left[1]
        self.bot_y = bottom_right[1]
        self.left_x = top_left[0]
        self.right_x = bottom_right[0]

        self.half_width = int((self.right_x-self.left_x)/2)
        self.half_height = int((self.bot_y-self.top_y)/2)
        self.center_x = self.left_x+self.half_width
        self.center_y = self.top_y+self.half_height

    # Define an obstacle by defining coordinates of center, and its width and height
    #   Obstacle( [5, 10], 10, 20)
    def __init__(self, center, width, height):
        self.half_width = int(width/2)
        self.half_height = int(height/2)
        self.center_x = center[0]
        self.center_y = center[1]

        self.top_y = self.center_y - self.half_height
        self.bot_y = self.center_y + self.half_height
        self.left_x = self.center_x - self.half_width
        self.right_x = self.center_x + self.half_width

    def __str__(self):
        return "Obstacle([" + str(self.left_x) + "," + str(self.top_y) + "],[" + \
               str(self.right_x) + "," + str(self.bot_y) + "])"
