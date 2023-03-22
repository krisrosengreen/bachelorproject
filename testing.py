class test:
    def __init__(self):
        self.n = 0

    def test(self):
        for i in range(100):
            self.n = i
            yield i

t = test()
