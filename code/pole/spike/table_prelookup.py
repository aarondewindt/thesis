

min_x = -6
max_x = 8
n = 8

x = 0.2

interval = (max_x - min_x) / (n - 1)
idx = round(x / interval) - min_x / interval

print(x, idx)

print(idx * (max_x - min_x) / (n - 1) + min_x)
