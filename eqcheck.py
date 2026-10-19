import numpy as np
from tsl_slt import Grid, Model
g1=Grid(1,4000); g2=Grid(2,200)
mods=[Model("w^2",g1,lambda w:w**2,1/2,1),Model("w^4",g1,lambda w:w**4,1/4,1),Model("w^6",g1,lambda w:w**6,1/6,1),
      Model("w1^2+w2^2",g2,lambda a,b:a**2+b**2,1,1),Model("(w1^2+w2^2)^2",g2,lambda a,b:(a**2+b**2)**2,1/2,1),
      Model("w1^2 w2^2",g2,lambda a,b:a**2*b**2,1/2,2),Model("w1^2+w2^4",g2,lambda a,b:a**2+b**4,3/4,1)]
for M in mods:
    hi = 1e4 if M.g.d==1 else 400
    ns=np.geomspace(hi/30,hi,12); F=np.array([-M.logZ(n) for n in ns])
    # 局所傾き dF/dlog n と、loglog を補正した傾き
    sl=np.gradient(F,np.log(ns)); corr=sl+(M.m-1)/np.log(ns)
    A=np.vstack([np.log(ns),np.log(np.log(ns)),np.ones_like(ns)]).T
    c,*_=np.linalg.lstsq(A,F,rcond=None)
    print(f"{M.name:16s} true lam={M.lam:.3f} m={M.m} | slope@n_max={sl[-1]:.3f} corrected={corr[-1]:.3f} | fit lam={c[0]:.3f}, -(m-1)={c[1]:.3f}")
