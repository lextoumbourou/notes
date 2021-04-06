class QuickFindUF(object):

    def __init__(self, N):
        self.ids = [i for i in range(N)]

    def connected(self, p, q):
        """Return True if objects connected."""
        return self.ids[p] == self.ids[q]

    def union(self, p, q):
        """Connect each element connected to p to q."""
        pid = self.ids[p]
        qid = self.ids[q]

        for i in range(len(self.ids)):
            if self.ids[i] == pid:
                self.ids[i] = qid


if __name__ == '__main__':
    qf = QuickFindUF(4)
    assert qf.ids == [0, 1, 2, 3]

    assert not qf.connected(0, 3)

    qf.union(0, 3)
    assert qf.ids == [3, 1, 2, 3]

    assert qf.connected(0, 3)
    
    qf.union(3, 1)
    assert qf.ids == [1, 1, 2, 1]

    assert qf.connected(3, 1)

    qf.union(0, 2)
    assert qf.ids == [2, 2, 2, 2]
