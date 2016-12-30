from mdu import weight_save, weight_load, id_save, id_load, batchIter
import os

sN = 10
sL = 50
qL = 15
batch_size = 128

delta = 0.01

def fake_weight(data):
    pass

def fake_ids(weight):
    fake = []
    for sen, que in weight:
        fs = [ [0.0]*len(_) for _ in sen ]
        fq = [0.0]*len(que)
        fake.append( [fs, fq, [[0]], [1,2,3]] )
    return fake

def test_save_load_feed_weight(data):
    fname = 'tmp_save_weight.txt'
    print 'saving'
    weight_save(fname, data)
    print 'loading'
    test = weight_load(fname)

    print 'conforming'
    for d, t in zip(data, test):
        ds, dq = d
        ts, tq = t
        assert len(ds) == len(ts)
        for i, j in zip(ds, ts):
            assert len(i) == len(j)

        assert len(dq) == len(tq)

    print 'feeding'
    ids = fake_ids(test)
    titer = batchIter(batch_size, ids, test,
                        sN, sL, qL)

    titer.next()
    for data in titer:
        pass

    print 'All good, cleaning'
    os.remove(fname)



