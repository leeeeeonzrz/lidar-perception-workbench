"""Observed geometric candidates only; unknown cells never count as free/flat."""
from dataclasses import dataclass
import numpy as np
from .geometry import clean_points

@dataclass(frozen=True)
class LandingConfig:
    resolution:float=0.5
    radius:float=1.5
    cell_std:float=0.08
    cell_range:float=0.20
    region_std:float=0.10
    minimum_points:int=4
    maximum_cells:int=1_000_000

def erode_disk(mask,disk):
    """Binary erosion with false borders; NumPy fallback keeps the core lightweight."""
    try:
        from scipy.ndimage import binary_erosion
        return binary_erosion(mask,structure=disk,border_value=0)
    except ImportError:
        radius=disk.shape[0]//2
        padded=np.pad(mask,radius,constant_values=False)
        output=np.ones_like(mask,dtype=bool)
        rows,cols=mask.shape
        for y,x in np.argwhere(disk):output &= padded[y:y+rows,x:x+cols]
        return output

def landing_candidate(points,config=LandingConfig(),origin=(0.0,0.0)):
    p=clean_points(points)
    if not len(p):return None
    parameters=[config.resolution,config.radius,config.cell_std,config.cell_range,config.region_std]
    if not np.isfinite(parameters).all() or min(parameters)<=0 or config.minimum_points<1:
        raise ValueError("invalid landing configuration")
    keys=np.floor(p[:,:2]/config.resolution).astype(np.int64)
    unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse);mean=np.bincount(inverse,weights=p[:,2])/count
    variance=np.maximum(0,np.bincount(inverse,weights=p[:,2]**2)/count-mean**2)
    low=np.full(len(unique),np.inf);high=np.full(len(unique),-np.inf)
    np.minimum.at(low,inverse,p[:,2]);np.maximum.at(high,inverse,p[:,2])
    minimum=unique.min(axis=0);width,height=map(int,unique.max(axis=0)-minimum+1)
    if width*height>config.maximum_cells:raise ValueError("grid allocation budget exceeded")
    flat=np.zeros((height,width),dtype=bool);heights=np.full((height,width),np.nan)
    x,y=(unique-minimum).T
    flat[y,x]=(count>=config.minimum_points)&(np.sqrt(variance)<=config.cell_std)&(high-low<=config.cell_range)
    heights[y,x]=mean
    radius=int(np.ceil(config.radius/config.resolution))
    if 2*radius+1>min(width,height):return None
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    disk=xx*xx+yy*yy<=radius*radius
    valid=erode_disk(flat,disk)
    candidates=[]
    for row,col in np.argwhere(valid):
        patch=heights[row-radius:row+radius+1,col-radius:col+radius+1][disk]
        roughness=float(patch.std())
        if roughness>config.region_std:continue
        xy=(np.array([col,row])+minimum+0.5)*config.resolution
        candidates.append(dict(center=[float(xy[0]),float(xy[1]),float(patch.mean())],
            radius=config.radius,roughness=roughness,distance=float(np.linalg.norm(xy-origin))))
    return min(candidates,key=lambda item:item["distance"]) if candidates else None
