import time


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        print("{0} took {1} seconds.".format(self.name, time.time() - self.start))
