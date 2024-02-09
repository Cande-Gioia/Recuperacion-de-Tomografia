from multiprocessing import Process, Manager
import numpy as np

def dothing(L, i):  # the managed list `L` passed explicitly.
    L[i].extend([1000, 2000])

N = 256
off = 128.0

a = [-off + 2*off*i/(N-1) for i in range(N)]

manager = Manager()
L = manager.list()
for i in range(8):
    L.append(manager.list([1, 2, 3]))
processes = []
for i in range(8):
    p = Process(target=dothing, args=(L,i))  # Passing the list
    p.start()
    processes.append(p)
for p in processes:
    p.join()

print(np.concatenate(L))
manager.shutdown()