import numpy as np


class Environment:
    def __init__(self, size):
        self.x = 0
        self.y = 0
        self.x_max = size[0] - 1
        self.y_max = size[1] - 1

    def state(self):
        return self.x, self.y

    def reset(self):
        self.x = 0
        self.y = 0

    def perform_action(self, action):
        if action == 0:
            self.y += 1
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y += -1
        elif action == 3:
            self.x += -1
        else:
            raise Exception(f"Invalid action '{action}'.")

        self.x = np.clip(self.x, 0, self.x_max)
        self.y = np.clip(self.y, 0, self.y_max)

        if self.y == 0:
            if self.x == self.x_max:
                return True, 0
            else:
                return True, -100
        else:
            return False, -1


