"""Optional local GICP tracking. No loop closure, global map or inertial preintegration."""
from collections import deque
import numpy as np

def imu_rotation_increment(previous,current,lidar_to_imu):
    """All matrices are 3x3: R_WI(t-1), R_WI(t), R_IL."""
    return lidar_to_imu.T @ previous.T @ current @ lidar_to_imu

class LocalGICP:
    def __init__(self,history=4,voxel=0.2,correspondence=0.8):
        import open3d as o3d
        if history<1 or voxel<=0 or correspondence<=0:raise ValueError("invalid GICP config")
        self.o3d=o3d;self.history=deque(maxlen=history)
        self.voxel,self.correspondence=voxel,correspondence
        self.pose=np.eye(4)
    def update(self,points,predicted_pose):
        o3d=self.o3d
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points,dtype=float)))
        cloud=cloud.voxel_down_sample(self.voxel)
        prediction=np.asarray(predicted_pose,dtype=float)
        if prediction.shape!=(4,4) or not np.isfinite(prediction).all():raise ValueError("invalid pose")
        if len(cloud.points)<6:return prediction.copy(),False
        accepted=False;pose=prediction.copy()
        if self.history:
            target=o3d.geometry.PointCloud()
            for saved in self.history:target+=saved
            target=target.voxel_down_sample(self.voxel)
            result=o3d.pipelines.registration.registration_generalized_icp(cloud,target,
                self.correspondence,prediction,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
            accepted=result.fitness>0.3 and result.inlier_rmse<self.correspondence*0.5
            if accepted:pose=result.transformation
        # Do not contaminate the local map with a failed registration.
        if not self.history or accepted:
            cloud.transform(pose);self.history.append(cloud)
        self.pose=pose
        return pose.copy(),accepted
