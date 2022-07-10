import pickle


# Instances of this class are unpicklable by default because of the lambda
class Foo:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.d = lambda x: a * x**2 + b * x + c

    def __repr__(self) -> str:
        return f"<Foo {self.a, self.b, self.c}>"

    def __reduce__(self):
        return (Foo, (self.a, self.b, self.c))


foo = Foo(1, 2, 3)

print(foo)
print(foo.d(3))

pickled_foo = pickle.dumps(foo)

new_foo = pickle.loads(pickled_foo)

print(new_foo)
print(new_foo.d(3))
