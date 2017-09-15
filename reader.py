import itertools
from abc import abstractmethod

class Reader:
    def __init__(self, readable):
        self._readable = readable
    
    def shuffle(self):
        self._readable.shuffle()
    
    @abstractmethod
    def read(self, *args):
        raise NotImplementedError('Method is not implemented')

class LimitedReader(Reader):
    def __init__(self, readable, limit):
        super().__init__(readable)
        self._limit = limit

    def read(self, *args):
        return itertools.islice(self._readable.read(*args), self._limit)

class FilteredReader(Reader):
    def __init__(self, readable, filter):
        super().__init__(readable) 
        self._filter = filter
        
    def read(self, *args):
        return itertools.filterfalse(lambda i: not self._filter(i), self._readable.read(*args))

class CycleReader(Reader):
    def __init__(self, readable):
        super().__init__(readable) 
    
    def read(self, *args):
        return itertools.cycle(self._readable.read(*args))

if __name__ == '__main__':
    class InputMock:
        def read(self):
            return range(10)
        
    assert(len(InputMock().read()) == 10)
    assert(len(list(LimitedReader(InputMock(), 2).read())) == 2)
    assert(len(list(FilteredReader(InputMock(), lambda x: x > 1).read())) == 8)
    assert(len(list(itertools.islice(CycleReader(InputMock()).read(), 42))) == 42)
