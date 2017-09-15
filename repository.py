import os
import hashlib
import shutil

class Repository:
    def __init__(self, location):
        self._location = location

    def resolve(self, name):
        h = hashlib.sha256(str.encode(name)).hexdigest()
        return os.path.join(self._location, h[0:2], h[2:4], h[4:])
    
    def build(self, prefix, sources):
        for src in sources:
            dst = self.resolve(src)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(prefix, src), dst)

if __name__ == '__main__':
    path = 'test/.repo'

    assert(not os.path.exists(path))
    r = Repository(path)
    sources = ['test/reg.csv', 'README.md']
    r.build('.', sources)
    for s in sources:
        assert(os.path.exists(r.resolve(s)))
    shutil.rmtree(path)
