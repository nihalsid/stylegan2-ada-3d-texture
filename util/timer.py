import time


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, _, value, traceback):
        print('[%s] Elapsed: %s' % (self.name, time.time() - self.tstart))
