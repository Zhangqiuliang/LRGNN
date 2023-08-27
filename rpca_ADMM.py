import time
from numpy import *
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool



def prox_l1(v,lambdat):
    return maximum(0, v - lambdat) - maximum(0, -v - lambdat)


def prox_matrix(v,lambdat,prox_f):
    
    U,S,V = svd(v,full_matrices=False) 
    S = S.reshape((len(S),1))
    pf = diagflat(prox_f(S,lambdat)) 
    return U.dot(pf).dot(V.conj())


def avg(*args): 
    N = len(args)
    x = 0
    for k in range(N):
        x = x + args[k]
    x = x/N
    return x


def objective(X_1, g_2, X_2, g_3, X_3):
    tmp = svd(X_3,compute_uv=0)
    tmp = tmp.reshape((len(tmp),1))
    return norm(X_1,'fro')**2 + g_2*norm(hstack(X_2),1) + g_3*norm(tmp,1) 


def rpcaADMM(data):

    pool = ThreadPool(processes=3) 
    N = 3       
    A = float_(data)   
    m,n = A.shape 

    g2_max = norm(hstack(A).T,inf) 
    g3_max = norm(A,2)
    g2 = 0.15*g2_max
    g3 = 0.15*g3_max
    MAX_ITER = 100
    ABSTOL   = 1e-4
    RELTOL   = 1e-2

    start = time.time()

    lambdap = 1.0
    rho = 1.0/lambdap

    X_1 = zeros((m,n))
    X_2 = zeros((m,n))
    X_3 = zeros((m,n))
    z   = zeros((m,N*n))
    U   = zeros((m,n))

    h = {}
    h['objval'] = zeros(MAX_ITER)
    h['r_norm'] = zeros(MAX_ITER)
    h['s_norm'] = zeros(MAX_ITER)
    h['eps_pri'] = zeros(MAX_ITER)
    h['eps_dual'] = zeros(MAX_ITER)

    def x1update(x,b,l):
        return (1.0/(1.0+l))*(x - b)
    def x2update(x,b,l,g,pl):
        return pl(x - b, l*g)
    def x3update(x,b,l,g,pl,pm):
        return pm(x - b, l*g, pl)

    def update(func,item):
        return list(map(func,[item]))[0] 

    for k in range(1):

        B = avg(X_1, X_2, X_3) - A/N + U
        async_X1 = pool.apply_async(update, (lambda x: x1update(x,B,lambdap), X_1)) 
        async_X2 = pool.apply_async(update, (lambda x: x2update(x,B,lambdap,g2,prox_l1), X_2))
        async_X3 = pool.apply_async(update, (lambda x: x3update(x,B,lambdap,g3,prox_l1,prox_matrix), X_3))

        X_1 = async_X1.get()
        X_2 = async_X2.get()
        X_3 = async_X3.get()
        # (for termination checks only)
        x = hstack([X_1,X_2,X_3])
        zold = z
        z = x + tile(-avg(X_1, X_2, X_3) + A*1.0/N, (1, N))
        U = B

        h['objval'][k]   = objective(X_1, g2, X_2, g3, X_3)
        h['r_norm'][k]   = norm(x - z,'fro')
        h['s_norm'][k]   = norm(-rho*(z - zold),'fro');
        h['eps_pri'][k]  = sqrt(m*n*N)*ABSTOL + RELTOL*maximum(norm(x,'fro'), norm(-z,'fro'));
        h['eps_dual'][k] = sqrt(m*n*N)*ABSTOL + RELTOL*sqrt(N)*norm(rho*U,'fro');

        if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
            break

    h['admm_toc'] = time.time() - start
    h['admm_iter'] = k
    h['X1_admm'] = X_1
    h['X2_admm'] = X_2
    h['X3_admm'] = X_3

    return h

