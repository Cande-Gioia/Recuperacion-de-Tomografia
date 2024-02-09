import numpy as np
from collections import defaultdict
from scipy import sparse

from multiprocessing import Process, Manager

def ri_GC(s):
    N = s[0]
    P = s[1]
    angles = np.arange(0, 180, 180/P)

    pi = np.pi
    cos = np.cos
    sin = np.sin

    manager = Manager()

    rot_mat = manager.list()
    for ang in angles:
        rot_mat.append(np.array([[cos(ang*pi/180), -sin(ang*pi/180)], 
                                 [sin(ang*pi/180), cos(ang*pi/180)]], 
                                 dtype=np.float32))
    
    def longitudes_a_indices(a):
        b = [0]
        b.extend(a)
        for i in range(1, len(b)):
            b[i] += b[i-1]
        return b

    off = (N//2 - 0.5) if N % 2 == 0 else N//2
    off2 = off*off

    X = manager.list([-off + 2*off*i/(N-1) for i in range(N)])
    Y = manager.list([-off + 2*off*i/(N-1) for i in range(N)])

    K_THREADS = 8
    L = [(N)//K_THREADS for i in range(K_THREADS - 1)]
    L.append(N - sum(L))
    I = manager.list(longitudes_a_indices(L))

    col_q = manager.list([[0]*(L[i]*N) for i in range(K_THREADS)])
    row_idx = manager.list([[] for i in range(K_THREADS)])
    vals = manager.list([[] for i in range(K_THREADS)])
    
    def thread_generar_matriz(n):
        a = 0
        for yi in Y[I[n]:I[n+1]]:
            for xi in X:
                if xi*xi + yi*yi >= off2:
                    col_q[n][a] = 0
                    a += 1
                    continue

                fila = defaultdict(lambda: 0)

                for r, rot in enumerate(rot_mat):
                    v = (rot @ [xi,yi])

                    x = v[0]+off
                    y = off-v[1]
                    xint = int(x)
                    yint = int(y)
                    xm = x - xint
                    ym = y - yint

                    if xm < 0.001:
                        fila[xint*P + r] += 1
                    elif xm > 0.999:
                        fila[(xint+1)*P + r] += 1
                    else:
                        fila[xint*P + r] += 1-xm
                        fila[(xint+1)*P + r] += xm

                fila_ordenada = sorted(fila.items())

                row_idx[n].extend([x[0] for x in fila_ordenada])
                vals[n].extend([x[1] for x in fila_ordenada])
                col_q[n][a] = len(fila)
                a += 1

    processes = []
    for i in range(8):
        p = Process(target=thread_generar_matriz, args=(i,))  # Passing the list
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    A = sparse.csc_array((np.concatenate(vals), np.concatenate(row_idx), longitudes_a_indices(np.concatenate(col_q))), shape=(N*P, N*N)) #esta es la matriz, notar que es dispersa (sino no alcanza la memoria)
    manager.shutdown()
    print('OK')

ri_GC((256,180))