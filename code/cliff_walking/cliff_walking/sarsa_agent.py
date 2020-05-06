from collections import defaultdict
from random import randint, random


class SarsaAgent:
    def __init__(self, alpha, gamma, eps, environment):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q = defaultdict(lambda: 0.0)
        self.environment = environment
        self.reward_sum = 0

    def greedy_action(self):
        state = self.environment.state()
        best_action = randint(0, 3)
        best_value = self.q[(state, best_action)]
        for i in range(4):
            value = self.q[(state, i)]
            if value > best_value:
                best_action = i
                best_value = value
        return best_action

    def policy(self):
        if random() < self.eps:
            return randint(0, 3)
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
        return self.reward_sum

