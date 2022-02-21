import networkx as nx
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import sparse

import numpy as np
import scipy
from scipy.sparse.linalg import spilu
import time

def rqi(A, M, v=None, s=0, k=2, eps=1e-4, maxiters=100, seed=0):
    """Rayleigh quotient iteration 
    Args:
        A: A sparse jax matrix. Should be Hermitian and close to psd (for cg)
        M: Preconditioner for A
        v: constraint vector.
        s: bound on the smallest eigenvalue
        k: optional ridge regularization.
        maxiters: maximum # iterations
        eps:  rqi tolerance
        seed: random seed

    Returns:
        U: eigenvectors of A corresponding to S (columns)
        S: smallest k eigenvalues of A."""
    key = jax.random.PRNGKey(seed)
    n = A.shape[0]
    
    #Initialize v to a unit vector
    if v == None:
        v = jnp.ones((n,1))/jnp.sqrt(n)
        
    I = sp.identity(n)
    I = sparse.BCOO.from_scipy_sparse(I)
    s = 0.
    As = A - s*I
    _matvec = lambda x: As@x
    matvec = jit(_matvec)
    u_0 = jax.random.normal(key, shape=(n,1))#?A randomly generated column?
    
    '''
    Solve Ax = b by conjugate gradient. Used for finding inverse of (A-sI) * u
    '''
    def Aspsolve(b, **kwargs):
        return jax.scipy.sparse.linalg.cg(matvec, b, M=M, **kwargs)[0]
 
    v_1 = Aspsolve(v)
    
    '''
    ?Calculating PAP*u_k that will be used to calculate the error
    for ending the Rayleigh Quotient Iteration. P = I-v*v^T
    
    args:
    u: the u_k, the k^th eigenvector of A?
    '''
    @jit
    def papu(u):
        pu = u - v@(v.T@u) #P*u = (I-v*v^T)*u = u-v*(v^T*u)
        Apu = A@pu #AP*u_k
        pApu = Apu - v@(v.T@Apu) #Plug in P= I-v*v^T
        return pApu
    
    '''
    Solve P_yk
    '''
    def py(u_k, w):
        #Calculate eigenvectors after the first eigenvector
        if len(w.shape) != 0:
            c, Aiu, Aivw = C(u_k, w)
            Py = Aiu + jnp.expand_dims(jnp.sum(Aivw * c.T,1),1)
        
        #Call Aspsolve to solve for P*y_k from P(A−sI)P*yk =uk−1
        else:
            u_p = Aspsolve(u_k) #u_p is u', find u' by solving u'*(A-sI)=u_(k-1)
            c = -v.T@u_p/(v.T@v_1).item() #(9) in PAP.pdf
            
            Py = u_p + c*v_1 #(6) in PAP.pdf
        return Py
    
    '''
    Solve for c = [c1 c2]^T for solving P_yk
    '''
    @jit
    def C(u_k, w):
        c = jnp.zeros(w.shape[1]+1)
        vw = jnp.concatenate([v,w],axis=1) #Construct [v, w] in (10)
        Aiu = Aspsolve(u_k) #Compute (A-sI)^(-1)*u_k in (10) and store in Aiu
        Aivw = Aspsolve(vw) #Compute (A-sI)^(-1)*[v w] in (10) and store in Aivw
        
        c = jnp.linalg.inv(vw.T@Aivw)@(-vw.T@Aiu) #c = ([v w]^T * Aivw)^(-1) * (-[v w]^T * Aiu)
        return c, Aiu, Aivw
    
    '''
    Solve eigenvectors of A by solving the eigenvectors of PAP?
    '''
    def _rqi(u_k, w=jnp.array(0)):
        s_k = (u_k.T@A@u_k).item() #s_k = u_k^T * A * u_k
        err = jnp.linalg.norm(papu(u_k) - s_k*u_k) #Error term for convergence of Rayleigh Quotient iteration
        errs = [err]
        i = 0
        
        #Run Rayleigh Quotient iteration until ∥PAP*uk − sk*uk ∥ < ε or reach max iteration number
        while (err > eps) and (i < maxiters):
            Py = py(u_k,w) #Calculate P*y_k by calling py
            
            u_k = Py / jnp.linalg.norm(Py) #Calculate u_k = P*y_k/∥P*y_k∥
            
            s_k = (u_k.T@A@u_k).item() #s_k = u_k^T*A*u_k
            err = jnp.linalg.norm(papu(u_k) - s_k*u_k) #Calculate ∥PAP*uk − sk*uk ∥
            
            errs.append(err)
            if i > 25 and errs[-1] < errs[-2]:
                break
            
            i+=1
            
        return u_k, s_k

    U = []
    w = []
    S = []
    u,s = _rqi(u_0)#Why we need to run this on u_0
    i = 1
    
    #Calculate the first k eigenvectors
    while len(U) < k:
        _, key = jax.random.split(key)
        u_0 = jax.random.normal(key, shape=(n,1))
        if s > eps:
            U.append(u)
            S.append(s)
            print('eigenvalue {}: {:.3f}'.format(i, s))
        if len(U) >= k:
            break
        w.append(u)
        u, s = _rqi(u_0,w=jnp.concatenate(w,axis=-1))
        i += 1
    return jnp.concatenate(U,axis=-1), jnp.array(S)

# create a random sparse graph as a testcase
N = 10000
p = 0.01
graph = nx.erdos_renyi_graph(N, p)
A = nx.laplacian_matrix(graph)
del(graph)
"""
# broken preconditioning
def spiluprec(A):
    #B = spilu(A)
    #L = sparse.BCOO.from_scipy_sparse(B.L)
    #U = sparse.BCOO.from_scipy_sparse(B.U)

    #Ux = lambda x: U@x
    #Lx = lambda x: L@x

    #Uinvx = lambda x: jax.scipy.sparse.linalg.cg(Ux, x)[0]
    #Linvx = lambda x: jax.scipy.sparse.linalg.cg(Lx, x)[0]
    #M = lambda x: Uinvx(Linvx(x))
    
    m = sparse.BCOO.from_scipy_sparse(sp.diags(np.reciprocal(A.diagonal())))
    return lambda x: m@x
M = spiluprec(A)
"""
M = None
A = sparse.BCOO.from_scipy_sparse(A)
I = sp.identity(N)
I = sparse.BCOO.from_scipy_sparse(I)
A = A+I

t0 = time.time()
U,S = rqi(A, s=0, eps=1e-3, M=M)
rqitime = time.time() - t0
print('rqi eigenvalues: ',jnp.round(S,3), 's: ', rqitime)

# validate true eigenvalues
def sorted_eig(A):
    w,v = jnp.linalg.eig(A)
    sidx = jnp.argsort(w)
    return w[sidx],v[:,sidx]

t0 = time.time()
A = A.todense()
v_s = jnp.ones((A.shape[0],1))/np.sqrt(A.shape[0])
I = jnp.eye(v_s.shape[0])
pa = A - v_s@(v_s.T@A)
pap = pa - (pa@v_s)@v_s.T
setuptime = time.time() - t0

t0 = time.time()
nw, nv = sorted_eig(pap)
nptime = time.time() - t0
print('np eigenvalues: ',jnp.round(nw[:3].real,3), 's: ', nptime+setuptime)

t0 = time.time()
sw, sv = scipy.linalg.eigh(pap)
sptime = time.time() - t0
print('sp eigenvalues: ',jnp.round(sw[:3].real,3), 's: ', sptime+setuptime)
