import numpy as np
import random
import math

MAX = 1000000

def init_centroids(position, K):
    center = np.mean(position, axis=0, keepdims=True)
    dist = np.zeros((position.shape[0],))
    dist += np.linalg.norm(position-center, ord=2, axis=1)
    i = np.argsort(dist[:K])
    centroids = np.copy(position[i])
    return centroids


def distance(position, centroids, ord=2):
    position = np.expand_dims(position, axis=1)  # (N, 1, D)
    centroids = np.expand_dims(centroids, axis=0) 
    d = position - centroids
    d = np.linalg.norm(d, ord=ord, axis=2)
    return d

def Kmeans(position, K, maxiter=100, ord=2):
    centroids = init_centroids(position, K)
    W = np.zeros_like(position)
    for _ in range(maxiter):
        N, _ = position.shape
        W_old = W
        W = np.zeros_like(position)
        ranking = distance(position, centroids, ord)
        W[np.arange(N), np.argmin(ranking, axis=1)] = 1
        if (np.all(W_old-W == 0)): break
        centroids = W.T@position / np.sum(W, axis=0, keepdims=True).T
    W = np.zeros_like(position)
    W[np.arange(N), np.argmin(distance(position, centroids, ord), axis=1)] = 1
    
    return W, centroids

    

def BKmeans(position, K, delta_max, c=0.15, f_1=1.1, f_2=1.05, MAX_EPOCHS = 100):
    N = position.shape[0]
    clusters, centroids = Kmeans(position, K)
    
    # 2 Set initial cluster sizes
    size = clusters.sum(axis=0)
    
    # 3 Set initial values of scale
    p_now = 0
    p_next = MAX
    
    # 4 Set initial values of the min and max cluster sizes
    n_min = 0
    n_max = N
    
    # 5 Define functions
    def removePoint(i, j):
        clusters[i][j] = 0
        centroids[j] = np.mean(clusters[:,j][:,None] * position, axis=0)
        size[j] += c - 1
    
    def penaltyNext(i, j):
        mask = size > size[j]
        if np.sum(mask.astype(int)) > 0: 
            diffs = centroids[mask] - position[i][None]  
            diffs = np.sum(diffs ** 2, axis=1)
            diffj = np.sum((centroids[j] - position[i]) ** 2)
            result = (diffs - diffj) / (size[mask] - size[j])
            return np.min(result)
        else:
            return MAX
    
    def addPoint(i, jdot, j):
        clusters[i][jdot] = 1
        centroids[jdot] = np.mean(clusters[:,jdot][:,None] * position, axis=0)
        size[jdot] += 1
        size[j] -= c

    
    epoch = 0
    while n_max - n_min > delta_max and epoch < MAX_EPOCHS:
        f = f_1 if epoch < MAX_EPOCHS / 2 else f_2
        for i in range(N):
            j = clusters[i].argmax().item()
            removePoint(i, j)
            xi = np.expand_dims(position[i], axis=0)

            temp = np.sum((centroids-xi)**2, axis=1) + p_now * size 
            jdot = np.argmin(temp).item()
            p_next_i = penaltyNext(i, j)
            if p_now < p_next_i < p_next:
                p_next = p_next_i    
            addPoint(i, jdot, j)

        n_min = np.min(size)
        n_max = np.max(size)
        p_now = f * p_next
        p_next = MAX 
        epoch += 1
    cls = {}
    for i in range(N):
        j = np.argmax(clusters[i]).item()
        if (j not in cls):
            cls[j] = [i]
        else:
            cls[j].append(i)
    return cls
        
    

def GSC(W, K, t, gamma=0.5, alpha=1):
    N = len(W)
    W = np.array(W)
    P = W / W.sum(axis=1)
    P_hat = gamma * P + (1-gamma)/N
    
    # 1 Compute the generalized Laplacian L
    P_hat_t = np.linalg.matrix_power(P_hat, t)
    mu = ((1/N * (np.ones((1, N)) @ P_hat_t)) ** alpha).flatten()
    eps = mu * P.sum(axis=1)
    N_mu = np.diag(mu)
    E = np.diag(eps)
    L = N_mu + E - (N_mu@P + P.T@N_mu)
    
    # 2 Compute the eigenvectors of Laplacian L
    eig, eigv = np.linalg.eigh(L)
    
    # 3 embedding for the vertexs
    position = eigv[:,:K]

    # 4 clustering using BKmeans
    clusters = BKmeans(position, K, 5)
    return clusters 

def clustering(cfg):
    '''
        input: 
            cfg: parameters and config
        output:
            clustering: cities per cluster
    '''
    N = cfg['N']
    K = cfg['K']
    T = cfg['T']
    gamma = cfg['gamma']
    alpha = cfg['alpha']
    
    D_matrix = np.array(cfg['C'])[1:, 1:] * 1.0
    depost_time = np.array(cfg['C'][0])[1:]
    time_execute = np.array(cfg['D'])
    if cfg['is_add_time_execute']:
        D_matrix += time_execute.reshape(1, time_execute.shape[0])

    if cfg['is_add_depost_time']:
        D_matrix += depost_time.reshape(depost_time.shape[0], 1)
        D_matrix += depost_time.reshape(1, depost_time.shape[0])
    
    D_matrix += np.eye(N)*1.0
    D_matrix[D_matrix == 0] += 1e-3
    W = 1 / D_matrix 
    W -= np.diag(np.diag(W))
    clusters = GSC(W, K, T, gamma, alpha)
    return clusters



def TSP(cluster, cfg, weight):
    cluster = [x+1 for x in cluster]
    cluster += [0]
    cluster.sort()
    N = len(cluster)
    W = weight[cluster][:,cluster]
    W[np.arange(W.shape[0]), np.arange(W.shape[0])] = MAX
    idx = np.argsort(W, axis=None)
    root = [0]*N; side = [0]*N
    tree = {}; graph = {}
    count = 1; edges = 0
    for id in idx:
        x = id // N
        y = id % N
        if (side[x] >= 2): continue
        if (side[y] >= 2): continue
       
        if (root[x] != 0 and root[y] != 0):
            if (root[x] == root[y]): continue
            temp = root[y]
            for i in tree[temp]:
                root[i] = root[x]
            tree[root[x]] += tree[temp]
            tree.pop(temp)
        elif (root[x] == 0 and root[y] == 0):
            root[x] = root[y] = count
            tree[count] = [x, y]
            count += 1
        else:
            tree[max(root[x], root[y])].append(x if root[x] < root[y] else y)
            root[x] = root[y] = max(root[x], root[y])
            
        if x not in graph: graph[x] = [y]
        else: graph[x].append(y) 
        if y not in graph: graph[y] = [x]
        else: graph[y].append(x)
        edges += 1
        side[x] += 1
        side[y] += 1
        if (edges == N): 
            break
        
    last = []
    for i in range(N):
        if side[i] == 1:
            last += [i]
            side[i] += 1
    graph[last[1]].append(last[0])
    graph[last[0]].append(last[1]) 
    curr = 0
    side[curr] += 1
    ne = graph[curr][0]
    solution = [0]
    while (ne != 0):
        solution += [ne]
        curr = ne
        side[curr] += 1
        ne = graph[curr][0] if side[graph[curr][0]] < 3 else graph[curr][1]
        if (side[ne] >= 3): break
    solution += [0]
    solution = [cluster[i] for i in solution]
    return solution
    
def cal_value(C, path):
    value = sum(C[path[i-1]][path[i]] for i in range(len(path)))
    return value

def cal_delta(C, j, k, schedule):
    if isinstance(schedule[0], int):
        n = len(schedule)
        return C[schedule[j-1]][schedule[k]] + C[schedule[k]][schedule[(j + 1)%n]] + \
               C[schedule[k-1]][schedule[j]] + C[schedule[j]][schedule[(k + 1)%n]] - \
               C[schedule[j-1]][schedule[j]] - C[schedule[j]][schedule[(j + 1)%n]] - \
               C[schedule[k-1]][schedule[k]] + C[schedule[k]][schedule[(k + 1)%n]]
    
    else:
        n = len(schedule[0])
        m = len(schedule[1])
        return C[schedule[0][j-1]][schedule[1][k]] + C[schedule[1][k]][schedule[0][(j + 1)%n]] - \
               C[schedule[0][j-1]][schedule[0][j]] - C[schedule[0][j]][schedule[0][(j + 1)%n]] , \
               C[schedule[1][k-1]][schedule[0][j]] + C[schedule[0][j]][schedule[1][(k + 1)%m]] - \
               C[schedule[1][k-1]][schedule[1][k]] + C[schedule[1][k]][schedule[1][(k + 1)%m]]

def localSearch(cfg, schedule, thresh=0.5, MAX_ITER=5000):
    thresh = 0.5
    idx = 0
    K = cfg['K']
    C = cfg['C']
    values = [cal_value(C, s) for s in schedule]
    while idx < MAX_ITER:
        i = idx % K
        p = random.random()
        delta = 0
        idx += 1
        if p < thresh:
            a, b = -1, -1
            n = len(schedule[i])
            for j in range(1, n-1):
                for k in range(1, n-1):
                    if j == k: continue
                    temp = cal_delta(C, j, k, schedule[i])
                    if delta > temp:
                        a, b = j, k
                        delta = temp
            if a != b:
                schedule[i][a], schedule[i][b] = schedule[i][b], schedule[i][a]
                values[i] += delta
        else:
            thresh *= math.exp2(-1/idx)
            a, b = -1, -1
            sto = [0, 0]
            o = random.randint(0, len(schedule)-1)
            n = len(schedule[i])
            m = len(schedule[o])
            for j in range(1, n-1):
                for k in range(1, m-1):
                    temp1, temp2 = cal_delta(C, j, k, (schedule[i], schedule[o]))
                    if delta > temp1 + temp2:
                        a, b = j, k
                        delta = temp1 + temp2
                        sto[0], sto[1] = temp1, temp2
            if a != -1:
                schedule[i][a], schedule[o][b] = schedule[o][b], schedule[i][a]
                values[i] += sto[0]
                values[o] += sto[1]            
        
    
    
def main(cfg):
    clusters = clustering(cfg)
    schedule = []
    weight = np.array(cfg['C'])
    for x in clusters:
        traveling = TSP(clusters[x], cfg, weight)
        schedule += [traveling]
    localSearch(cfg, schedule)
    return schedule