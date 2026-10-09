import numpy as np

def scene(seed=0):
    rng=np.random.default_rng(seed)
    x,y=np.meshgrid(np.arange(-5,5,0.1),np.arange(-5,5,0.1))
    ground=np.column_stack([x.ravel(),y.ravel(),rng.normal(0,0.01,x.size)])
    obstacle=rng.uniform([2,2,0.2],[3,3,1.8],size=(600,3))
    return np.vstack([ground,obstacle])
