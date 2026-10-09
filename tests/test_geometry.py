import unittest
import numpy as np
from lidar_workbench.geometry import voxel_downsample,ground_plane,dbscan
from lidar_workbench.landing import landing_candidate,LandingConfig
from lidar_workbench.pipeline import ResultJoiner

class GeometryChecks(unittest.TestCase):
    def test_voxel_centroid(self):
        result=voxel_downsample([[0,0,0],[0.2,0,0],[2,0,0]],1)
        self.assertEqual(len(result),2);self.assertTrue(np.allclose(result[0],[0.1,0,0]))
    def test_ground_and_clusters(self):
        x,y=np.meshgrid(np.linspace(-2,2,12),np.linspace(-2,2,12))
        p=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        mask,plane=ground_plane(p)
        self.assertTrue(mask.all());self.assertAlmostEqual(abs(plane[2]),1)
        labels=dbscan([[0,0,0],[0.1,0,0],[3,0,0],[3.1,0,0]],0.2,2)
        self.assertEqual(len(set(labels)),2)
    def test_unknown_hole_is_not_flat(self):
        x,y=np.meshgrid(np.arange(-2,2,0.1),np.arange(-2,2,0.1))
        p=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        config=LandingConfig(radius=1.5)
        self.assertIsNotNone(landing_candidate(p,config))
        # A wide unobserved cross prevents any full disk from being supported.
        p=p[(abs(p[:,0])>0.6)&(abs(p[:,1])>0.6)]
        self.assertIsNone(landing_candidate(p,config))
    def test_results_do_not_cross_frames(self):
        joiner=ResultJoiner()
        self.assertIsNone(joiner.add(dict(sequence=1,kind="obstacles"),now=0))
        self.assertIsNone(joiner.add(dict(sequence=2,kind="landing"),now=0))
        self.assertIsNotNone(joiner.add(dict(sequence=1,kind="landing"),now=0))
if __name__=="__main__":unittest.main()
