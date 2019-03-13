from enum import IntEnum
from random import randint, random
from collections import defaultdict


def main():
    agent = BasicAgent(
        0.1,

    )


class Action(IntEnum):
    """
    Possible actions.
    """
    left = 0
    right = 1


class BasicAgent:
    """
    Simple agent with no temporal difference. Because of this it can only learn in
    an environment of size 3.
    """
    def __init__(self, eps: float, gamma: float, alpha: float, environment: 'Environment'):
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.q_table = defaultdict(float)
        self.environment = environment

    def print_greedy_action(self):
        """
        Prints the greedy action taken by the agent at each position.
        """
        world = ["x"] + [" "] * (self.environment.size - 2) + ["o"]
        for pos in range(1, self.environment.size):
            world[pos] = ["←", "→"][self.q_table[pos, Action.left] < self.q_table[pos, Action.right]]
        print("".join(world))

    def run_iteration(self):
        """
        Runs a single iteration.
        """
        # Get state at the beginning of the iteration.
        x_0 = self.environment.agent_position

        # Choose action
        if self.eps > random():
            action = Action(randint(0, 1))
        else:
            action = Action(self.q_table[x_0, Action.left] < self.q_table[x_0, Action.right])

        # Perform action
        reward, is_terminal = self.environment.perform_action(action)

        # Learn
        # TODO: implement TD here.
        self.q_table[x_0, action] = reward

        return reward, is_terminal

    def run_episode(self, n_max):
        reward_sum = 0
        for i in range(n_max):
            reward, is_terminal = self.run_iteration()
            reward_sum += abs(reward)
            if is_terminal:
                break
        return reward_sum


class Environment:
    def __init__(self, size: int):
        self.size = size
        self.agent_position = randint(0, size - 2)

    def print(self):
        world = ["x"] + [" "] * (self.size - 2) + ["o"]
        world[self.agent_position] = "a"
        print("".join(world))

    def reset(self):
        self.agent_position = randint(1, self.size - 2)

    def perform_action(self, action: Action):
        # Perform the action
        if action is Action.right:
            self.agent_position += 1
        else:
            self.agent_position -= 1

        # Determine reward
        reward = 0
        is_terminal = False
        if self.agent_position <= 0:
            reward = -1
            is_terminal = True
        elif self.agent_position >= (self.size - 1):
            reward = 1
            is_terminal = True

        return reward, is_terminal


if __name__ == '__main__':
    main()
