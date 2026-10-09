"""ROS 1 adapter for an already gravity-aligned PointCloud2 stream."""
import time
import numpy as np
from lidar_workbench.pipeline import Frame,PerceptionPipeline

def main():
    import rospy
    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2
    rospy.init_node("perception_workbench")
    pipeline=PerceptionPipeline();pipeline.start()
    def receive(message):
        points=np.asarray(list(pc2.read_points(message,field_names=("x","y","z"),skip_nans=True)),dtype=float)
        if not points.size:return
        pipeline.submit(Frame(message.header.seq,message.header.stamp.to_sec(),points))
    subscriber=rospy.Subscriber(rospy.get_param("~cloud_topic","points_aligned"),PointCloud2,receive,queue_size=1)
    rate=rospy.Rate(20)
    try:
        while not rospy.is_shutdown():
            for result in pipeline.poll():
                rospy.loginfo("frame=%s kind=%s error=%s",result["sequence"],result["kind"],result["error"])
            rate.sleep()
    finally:pipeline.close()
if __name__=="__main__":main()
