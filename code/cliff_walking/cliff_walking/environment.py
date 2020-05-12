from enum import IntEnum
import numpy as np


class Action(IntEnum):
    up = 0
    right = 1
    down = 2
    left = 3


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

    def actions(self, state=None):
        state = state or (self.x, self.y)
        actions = [Action.up,
                   Action.right,
                   Action.down,
                   Action.left]

        if state[1] <= 0:
            actions.remove(Action.down)
        elif state[1] >= self.y_max:
            actions.remove(Action.up)

        if state[0] <= 0:
            actions.remove(Action.left)
        elif state[0] >= self.x_max:
            actions.remove(Action.right)

        return actions

    def perform_action(self, action: Action):
        if action == Action.up:
            self.y += 1
        elif action == Action.right:
            self.x += 1
        elif action == Action.down:
            self.y += -1
        elif action == Action.left:
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


