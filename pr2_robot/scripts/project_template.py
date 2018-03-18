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

MODEL_PATH = '/home/robond/catkin_ws/model.sav'

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

def statistical_outlier_filter(cloud_data, mean=50, deviation=1.0):
	outlier_filter = cloud_data.make_statistical_outlier_filter()
	outlier_filter.set_mean_k(mean)
	outlier_filter.set_std_dev_mul_thresh(deviation)
	cloud_filtered = outlier_filter.filter()
	return cloud_filtered

# The VoxelGrid class that we’re about to present creates a 3D voxel grid (think about a 
# voxel grid as a set of tiny 3D boxes in space) over the input point cloud data. Then, in 
# each voxel (i.e., 3D box), all the points present will be approximated (i.e., downsampled) 
# with their centroid. This approach is a bit slower than approximating them with the center 
# of the voxel, but it represents the underlying surface more accurately. 
def voxel_grid_downsampling(cloud_data,LEAF_SIZE = 0.01):
    vox = cloud_data.make_voxel_grid_filter()
    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()
    return cloud_filtered

# The passthrough filter cut off values that are either inside or outside a given user range
# Here, i defined a function capable of choose the axis and the range for the filter
def Passthrough_filter(cloud_data,axis_min = 0.6,axis_max = 1.1,filter_axis = 'z'):
    # Create a PassThrough filter object.
    passthrough = cloud_data.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
    return cloud_filtered

# The RANSAC algorithm assumes that all of the data we are looking at is comprised of both 
# inliers and outliers. Inliers can be explained by a model with a particular set of parameter 
# values, while outliers do not fit that model in any circumstance.
# The input to the RANSAC algorithm is a set of observed data values, a parameterized model which 
# can explain or be fitted to the observations, and some confidence parameters.
# RANSAC achieves its goal by iteratively selecting a random subset of the original data. These data 
# are hypothetical inliers and this hypothesis is then tested as follows:
# 	1.) A model is fitted to the hypothetical inliers, i.e. all free parameters of the model are 
# reconstructed from the inliers.
# 	2.) All other data are then tested against the fitted model and, if a point fits well to the estimated 
# model, also considered as a hypothetical inlier.
# 	3.) The estimated model is reasonably good if sufficiently many points have been classified as hypothetical inliers.
# 	4.) The model is reestimated from all hypothetical inliers, because it has only been estimated from 
# the initial set of hypothetical inliers.
# 	5.)Finally, the model is evaluated by estimating the error of the inliers relative to the model.
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

# Here we are creating a EuclideanClusterExtraction with point type PointXYZ since our point cloud is of type PointXYZ.
# We are also setting the parameters and variables for the extraction. Be careful setting the right value for setClusterTolerance(). 
# If you take a very small value, it can happen that an actual object can be seen as multiple clusters. On the other hand, if you set 
# the value too high, it could happen, that multiple objects are seen as one cluster. So our recommendation is to just test and try 
# out which value suits your dataset.
# We impose that the clusters found must have at least setMinClusterSize() points and maximum setMaxClusterSize() points.
# Now we extracted the clusters out of our point cloud and saved the indices in cluster_indices. To separate each cluster out of the 
# vector<PointIndices> we have to iterate through cluster_indices, create a new PointCloud for each entry and write all points of 
# the current cluster in the PointCloud. http://www.pointclouds.org/documentation/tutorials/cluster_extraction.php
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
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature_vector.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = pcl_cluster_msg
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)


    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# The object class will allow us to create an object (already segmentared) and save 
# the characteristic for each one of them. 

# The Object: the object we suppose to detect and grasp
# Object Name (Obtained from the pick list): The label name predicted for the object
# Arm Name (Based on the group of an object): the arm who is suppose to pick the object
# Group: where the object is supposed to be delivered
# Pick Pose (Centroid of the recognized object):  where the centroid of the object is
# Place pose: where the destinity box is 
class PickObject:

  	def __init__(self, object):
	    self.name = String()
	    self.arm = String()
	    self.pick_pose = Pose()
	    self.place_pose =  Pose()
	    self.name.data = str(object.label)
	    self.group = None 
	    self.yaml_dictonary = None
	    points = ros_to_pcl(object.cloud).to_array()
# To calculated the centroid we will take the mean of the points
	    x, y, z = np.mean(points, axis = 0)[:3]
	    self.pick_pose.position.x = np.asscalar(x) 
	    self.pick_pose.position.y = np.asscalar(y)
	    self.pick_pose.position.z = np.asscalar(z)
# The setter function. It receives the pick list and find the group where it belong and
# the place position where it will be delivered.
	def place(self, pick_list, dropbox_list):
  		for obj in pick_list:
			if obj['name'] == self.name.data:
				self.group = obj['group']
		        break
  		for box in dropbox_list:
		    if box['group'] == self.group:
		      	self.arm.data = box['name']
		        x, y, z = box['position']
		        self.place_pose.position.x = np.float(x) 
		        self.place_pose.position.y = np.float(y)
		        self.place_pose.position.z = np.float(z)        
		        break
# This function just allow to call the helper function to create an dictionary with all the features
# of every object
	def Make_yaml_dict(self, scene):
	    self.yaml_dictonary = make_yaml_dict(scene, self.arm, self.name, self.pick_pose, self.place_pose)
	  

# function to load parameters and request PickPlace service
def pr2_mover(object_list):
  	test_scene = Int32()
  	test_scene.data = 2
  	file = []
  	# Get information from the YAML files
  	pick_list = rospy.get_param('/object_list')
  	dropbox_list = rospy.get_param('/dropbox')
	for Object in object_list:
		pickObject = PickObject(Object)
		pickObject.place(pick_list, dropbox_list)
		pickObject.Make_yaml_dict(test_scene)
		file.append(pickObject.yaml_dictonary)
		#print(pickObject.yaml_dictonary)
		rospy.wait_for_service('pick_place_routine')
	rospy.wait_for_service('pick_place_routine')
	send_to_yaml("output_model_" + str(2) + '.yaml', file)

if __name__ == '__main__':
	# TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_clusters", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    # TODO: Load Model From disk
    # Load Model From disk
    model = pickle.load(open(MODEL_PATH, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
		rospy.spin()