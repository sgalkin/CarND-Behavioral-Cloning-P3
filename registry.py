import operator
import copy
import os
import csv

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class Registry:
    CENTER = 0
    LEFT = 1
    RIGHT = 2
    STEERING = 3

    def __init__(self, path):
        def get_dialect(sample):
            dialect = csv.Sniffer().sniff(sample)
            dialect.skipinitialspace = True
            dialect.strict = True
            return dialect

        def read_header(file, sample):
            return next(csv.reader(file)) if csv.Sniffer().has_header(sample) else None

        self.prefix = os.path.dirname(path)
        
        with open(path, 'r') as f:
            sample = f.read(4096)
            f.seek(0)
        
            self.header = read_header(f, sample)
            self.dialect = get_dialect(sample)
            self._data = list(csv.reader(f, dialect=self.dialect))
                    
    def __len__(self):
        return len(self._data)
            
    def shuffle(self):
        self._data = shuffle(self._data)
        
    def read(self, *args):
        getter = operator.itemgetter(*args) if args else None
        for l in self._data:
            yield getter(l) if getter else l

    def split(self, fraction):
        data = self._data
        self._data = []
        first = copy.deepcopy(self)
        second = copy.deepcopy(self)
        self._data = shuffle(data)
        
        first._data, second._data = train_test_split(self._data, test_size=fraction)
        return first, second

    def resolve(self, name):
        return os.path.join(self.prefix, name)

    def store(self, name):
        with open(name, 'w') as csvfile:
            writer = csv.writer(csvfile, dialect=self.dialect)
            writer.writerows(self._data)
    
if __name__ == '__main__':
    r = Registry('test/reg.csv')
    assert(len(r) == 9)

    r.shuffle()
    reg_lst = [l for l in r.read() ]
    lst = [l for l in Registry('test/reg.csv').read() ]
    assert(reg_lst != lst)

    r1, r2 = r.split(0.5)
    assert(abs(len(r1) - len(r2)) <= 1) 

    r1.store('test/reg1.csv')
