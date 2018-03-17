#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    #print(yaml_dict)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

#def statistical_outlier_filter(cloud_data, mean=50, deviation=1.0):
#	outlier_filter = cloud_data.make_statistical_outlier_filter()
#	outlier_filter.set_mean_k(mean)
#	outlier_filter.set_std_dev_mul_thresh(deviation)
#	cloud_filtered = outlier_filter.filter()
#	return cloud_filtered

def voxel_grid_downsampling(cloud_data,LEAF_SIZE = 0.01):
    vox = cloud_data.make_voxel_grid_filter()
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    return cloud_filtered

def Passthrough_filter(cloud_data,axis_min = 0.6,axis_max = 1.1,filter_axis = 'z'):
    # Create a PassThrough filter object.
    passthrough = cloud_data.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
    return cloud_filtered

def RANSAC(cloud_data,max_distance = 0.01):
    seg = cloud_data.make_segmenter()
    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    # Max distance for a point to be considered fitting the model
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    cloud_table = cloud_data.extract(inliers, negative=False)
    cloud_objects = cloud_data.extract(inliers, negative=True)
    return cloud_table,cloud_objects

def Euclidean_clustering(cloud_objects,cluster_tolerance=0.02, min_cluster_size=30, max_cluster_size=40000):
    # Apply function to convert XYZRGB to XYZ
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(cluster_tolerance)
    ec.set_MinClusterSize(min_cluster_size)
    ec.set_MaxClusterSize(max_cluster_size)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return white_cloud, cluster_indices, cluster_cloud

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
# Exercise-2 TODOs:
    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)    
	cloud_filtered = voxel_grid_downsampling(pcl_data)
    #Passthrough filter implementation
    cloud_filtered=Passthrough_filter(cloud_filtered)
    cloud_filtered=Passthrough_filter(cloud_filtered,axis_min = 0.42,axis_max = 1.55,filter_axis = 'x')
    # TODO: RANSAC Plane Segmentation
    cloud_table,cloud_objects=RANSAC(cloud_filtered,max_distance = 0.01)
    # TODO: Euclidean Clustering
    white_cloud, cluster_indices, cluster_cloud = Euclidean_clustering(cloud_objects)
    # TODO: Convert PCL data to ROS messages
    pcl_msg_table = pcl_to_ros(cloud_table)
    pcl_msg_object = pcl_to_ros(cloud_objects)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    # TODO: Publish ROS messages
    pcl_objects_pub.publish(pcl_msg_object)
    pcl_table_pub.publish(pcl_msg_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
# Exercise-3 TODOs:
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
    	# Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        pcl_cluster_msg = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector
        color_hist = compute_color_histograms(pcl_cluster_msg, using_hsv=True)
        normals = get_normals(pcl_cluster_msg)
        normal_hist = compute_normal_histograms(normals)
        feature_vector = np.concatenate((color_hist, normal_hist))

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass



# function to load parameters and request PickPlace service
def pr2_mover(object_list):
  	


if __name__ == '__main__':

    

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
		rospy.spin()