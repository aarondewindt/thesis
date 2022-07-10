import pickle
from numba.experimental import jitclass


@jitclass
class Counter:
    value: int

    def __init__(self):
        self.value = 0

    def get(self) -> int:
        ret = self.value
        self.value += 1
        return ret

    def __reduce__(self):
        return Counter, tuple()


counter = Counter()

print(counter.get())
print(counter.get())
print(counter.get())


pickled_counter = pickle.dumps(counter)
