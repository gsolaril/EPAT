import sys, time

class ProgBar:
    def __init__(self, steps, width = 100, char = "■"):
        self.steps = steps ; self.char = char
        self.width = width ; self.times = [0]
        self._reset()
    def _bar(self, X):
        self.times.append(time.time() - self.clock)
        if (len(self.times) > 20): self.times = self.times[-21:]
        line = self.char*int(self.width*X) + " "*int(self.width*(1 - X))
        left = (self.steps - self.prog)*sum(self.times)/len(self.times)
        line = "\r[%s] %d%% (%d secs left)" % (line, 100*X, left)
        sys.stdout.write(line)  ;  sys.stdout.flush()
        self.clock = time.time()
    def _reset(self):
        self.clock = time.time()
        self.prog = 0
        self._bar(X = 0)
        sys.stdout.flush()
    def up(self, n = 1):
        self.prog = min(self.steps, self.prog + n)
        self._bar(X = self.prog/self.steps)