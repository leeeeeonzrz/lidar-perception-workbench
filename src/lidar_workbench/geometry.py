"""All coordinates are meters in a gravity-aligned, z-up frame."""
from collections import deque
import numpy as np

def clean_points(points):
    array=np.asarray(points,dtype=float)
    if array.ndim!=2 or array.shape[1]!=3: raise ValueError("expected Nx3 points")
    return array[np.isfinite(array).all(axis=1)]

def voxel_downsample(points,size):
    if not np.isfinite(size) or size<=0: raise ValueError("invalid voxel size")
    p=clean_points(points)
    if not len(p): return p
    keys=np.floor(p/size).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse)
    return np.column_stack([np.bincount(inverse,weights=p[:,axis])/count for axis in range(3)])

def ground_plane(points,threshold=0.08,iterations=80,max_slope_deg=25,seed=0):
    p=clean_points(points)
    if not np.isfinite(threshold) or threshold<=0 or iterations<1:raise ValueError("invalid RANSAC budget")
    if not 0<=max_slope_deg<90:raise ValueError("invalid slope limit")
    if len(p)<3:return np.zeros(len(p),dtype=bool),None
    rng=np.random.default_rng(seed);best=np.zeros(len(p),dtype=bool);plane=None
    minimum_vertical=np.cos(np.deg2rad(max_slope_deg))
    for _ in range(iterations):
        a,b,c=p[rng.choice(len(p),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-10:continue
        normal/=length
        if abs(normal[2])<minimum_vertical:continue
        distance=-np.dot(normal,a)
        mask=np.abs(p@normal+distance)<=threshold
        if mask.sum()>best.sum():best,plane=mask,np.r_[normal,distance]
    if plane is not None and best.sum()>=3:
        center=p[best].mean(axis=0)
        _,_,right=np.linalg.svd(p[best]-center,full_matrices=False)
        normal=right[-1]
        if normal[2]<0:normal=-normal
        if normal[2]>=minimum_vertical:
            plane=np.r_[normal,-normal@center]
            best=np.abs(p@plane[:3]+plane[3])<=threshold
    return best,plane

def dbscan(points,eps=0.45,min_points=6):
    p=clean_points(points)
    if not np.isfinite(eps) or eps<=0 or min_points<1:raise ValueError("invalid clustering parameters")
    labels=np.full(len(p),-1,dtype=int)
    if not len(p):return labels
    try:
        from scipy.spatial import cKDTree
        tree=cKDTree(p)
        neighbors_of=lambda index:tree.query_ball_point(p[index],eps)
    except ImportError:
        # Portable reference path; SciPy is preferred for larger clouds.
        neighbors_of=lambda index:np.flatnonzero(np.sum((p-p[index])**2,axis=1)<=eps**2).tolist()
    visited=np.zeros(len(p),dtype=bool);cluster=0
    for seed in range(len(p)):
        if visited[seed]:continue
        visited[seed]=True;neighbors=neighbors_of(seed)
        if len(neighbors)<min_points:continue
        labels[seed]=cluster;queue=deque(neighbors);queued=set(neighbors)
        while queue:
            index=queue.popleft()
            if not visited[index]:
                visited[index]=True;near=neighbors_of(index)
                if len(near)>=min_points:
                    for candidate in near:
                        if candidate not in queued:queued.add(candidate);queue.append(candidate)
            if labels[index]<0:labels[index]=cluster
        cluster+=1
    return labels

def describe_cluster(points):
    p=clean_points(points)
    if len(p)<3:raise ValueError("cluster too small")
    center=p.mean(axis=0);xy=p[:,:2]-center[:2]
    values,vectors=np.linalg.eigh(xy.T@xy/len(p));axes=vectors[:,::-1]
    horizontal=xy@axes;low,high=horizontal.min(0),horizontal.max(0)
    rotation=np.eye(3);rotation[:2,:2]=axes
    if np.linalg.det(rotation)<0:rotation[:,1]*=-1
    # Box extents are sign invariant; center is transformed using the original 2-D basis.
    box_center=np.r_[center[:2]+axes@((low+high)/2),(p[:,2].min()+p[:,2].max())/2]
    extent=np.r_[high-low,np.ptp(p[:,2])]
    height=float(extent[2]);width=float(max(extent[0],extent[1]))
    label="slender" if height>max(0.5,3*width) else "extended"
    return dict(center=box_center.tolist(),extent=extent.tolist(),rotation=rotation.tolist(),
                category=label,point_count=len(p))

def obstacle_pipeline(points):
    p=voxel_downsample(points,0.15)
    mask,plane=ground_plane(p)
    elevated=p[~mask];labels=dbscan(elevated)
    clusters=[describe_cluster(elevated[labels==label]) for label in np.unique(labels) if label>=0]
    return dict(points=elevated,labels=labels,clusters=clusters,
                plane=None if plane is None else plane.tolist())
