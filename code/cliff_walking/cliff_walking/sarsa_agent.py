from collections import defaultdict
from random import random, sample

from cliff_walking.environment import Action


action_print_map = {
    Action.up: "↑",
    Action.right: "→",
    Action.down: "↓",
    Action.left: "←",
}


class SarsaAgent:
    def __init__(self, alpha, gamma, eps, environment):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = defaultdict(lambda: 0.0)
        self.environment = environment
        self.reward_sum = 0
        self.total_reward_sum = 0

    def greedy_action(self, state=None):
        state = state or self.environment.state()
        actions = self.environment.actions(state)
        best_action = actions[0]
        best_value = self.q[(state, best_action)]
        for action in actions[1:]:
            value = self.q[(state, action)]
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def print_greedy_policy(self):
        for y in reversed(range(self.environment.y_max + 1)):
            for x in range(self.environment.x_max + 1):
                action = self.greedy_action((x, y))
                print(f"{action_print_map[action]} ", end="")
            print("")

    def policy(self):
        if random() < self.eps:
            return sample(self.environment.actions(), 1)[0]
        else:
            return self.greedy_action()

    def step(self):
        s_t = self.environment.state()
        a_t = self.policy()
        is_terminal, r_t1 = self.environment.perform_action(a_t)

        value = self.q[(s_t, a_t)]
        s_t1 = self.environment.state()
        a_t1 = self.policy()
        self.q[(s_t, a_t)] = value + self.alpha * (r_t1 + self.gamma * self.q[(s_t1, a_t1)] - value)

        self.reward_sum += r_t1
        if is_terminal:
            self.q[(s_t1, a_t1)] = 0.

        return is_terminal

    def run(self, n):
        self.reward_sum = 0
        self.environment.reset()
        for i in range(n):
            if self.step():
                break
        self.total_reward_sum += self.reward_sum
        return self.reward_sum

