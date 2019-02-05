from math import ceil

n_episodes = 901
n_record = 34

n_recorded = 0

for i in range(n_episodes):
    record = int(i % ((n_episodes / n_record) + 1)) == 0 or (i == (n_episodes - 1))
    print(i, record)
    if record:
        n_recorded += 1

print(n_recorded)
