import open3d as o3d
import numpy as np
import hdbscan  # Import HDBSCAN
import matplotlib.pyplot as plt
import json
import os
import time

# Parameters
ground_min = -2.5  # Adjust based on your dataset's ground level
ground_max = -1.5  # Z-value range for ground points
min_cluster_size = 10  # Minimum size of clusters for HDBSCAN
min_samples = 5        # Minimum number of samples in a cluster

# Define the folder containing your PCD files
pcd_folder = "IDD3D/20220118103308_seq_22/lidar/"
label_folder = "IDD3D/20220118103308_seq_22/label/"
output_folder = "IDD3D/20220118103308_seq_22/lidar/output_images/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

pcd_files = [f for f in os.listdir(pcd_folder) if f.endswith(".pcd")]
pcd_files = sorted(pcd_files)

# Initialize the Visualizer once
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Ground vs Non-Ground Points", width=800, height=600)

def load_labels_from_json(json_file_path):
    with open(json_file_path, "r") as file:
        return json.load(file)

# Process each PCD file
for i, pcd_file in enumerate(pcd_files):
    try:
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_folder, pcd_file))
        points = np.asarray(pcd.points)
        points = points[(points[:, 1] >= -30) & (points[:, 1] <= 50)]  # Filter by y-range
    
        # Step 1: Ground Segmentation
        ground_mask = (points[:, 2] >= ground_min) & (points[:, 2] <= ground_max)
        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]
    
        # Create separate PointCloud objects for ground and non-ground points
        ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd = o3d.geometry.PointCloud()
    
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
        ground_pcd.colors = o3d.utility.Vector3dVector(np.ones((ground_points.shape[0], 3)) * [255, 255, 237])  # Grey color
    
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
    
        # Step 2: HDBSCAN Clustering on Non-Ground Points
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(non_ground_points)
    
        # Generate unique colors for clusters
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        print(f"Number of clusters found: {num_clusters}")
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, num_clusters))
    
        # Assign colors to clusters
        cluster_colors = np.zeros((non_ground_points.shape[0], 3))  # Default to black for noise
        for cluster_id in unique_labels:
            if cluster_id == -1:
                continue  # Noise remains black
            cluster_color = colors[cluster_id % len(colors)][:3]  # Use RGB from colormap
            cluster_colors[labels == cluster_id] = cluster_color
    
        # Apply cluster colors to non-ground points
        non_ground_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

        fname = f"{pcd_file.split('.')[0]}.json"
        json_file_path = os.path.join(label_folder, fname)  # Replace with your actual JSON file path

        # Load the labels from the JSON file
        labels_data = load_labels_from_json(json_file_path)
    
        # Step 3_1: Filter Labels Based on x-Position Range
        filtered_labels = []
        for label in labels_data:
            position = np.array([label["psr"]["position"]["x"], label["psr"]["position"]["y"], label["psr"]["position"]["z"]])
            if -30 <= position[1] <= 50:  # Adjust as needed based on filtering criteria for y-axis
                filtered_labels.append(label)
    
        # Step 4: Create Bounding Boxes from Labels
        bounding_boxes = []
        label_texts = []
        for label in filtered_labels:
            position = np.array([label["psr"]["position"]["x"], label["psr"]["position"]["y"], label["psr"]["position"]["z"]])
            rotation = np.array([label["psr"]["rotation"]["x"], label["psr"]["rotation"]["y"], label["psr"]["rotation"]["z"]])
            scale = np.array([label["psr"]["scale"]["x"], label["psr"]["scale"]["y"], label["psr"]["scale"]["z"]])
    
            # Create a 3D bounding box
            bbox = o3d.geometry.OrientedBoundingBox(center=position,
                                                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(rotation),
                                                    extent=scale)
            bbox.color = [1, 0, 0]  # White color for bounding box
    
            bounding_boxes.append(bbox)
            label_texts.append(label["obj_type"])
            
        
        # Visualization and saving code remains the same...
        # Clear previous geometries and add new ones for the current frame
        vis.clear_geometries()
        vis.add_geometry(ground_pcd)
        vis.add_geometry(non_ground_pcd)

        # Add bounding boxes
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)
        # Set render options for a professional look
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0, 0, 0])  # Black background
        render_option.point_size = 1.5  # Larger points for better visibility
        render_option.show_coordinate_frame = False  # Disable coordinate frame
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_params.extrinsic = np.array([
        [1, 0, 0, 0],  # Move the camera to the right (X-axis)
        [0, 0, -1, 10], # Move the camera slightly higher along the Z-axis for a better view
        [0, 1, 0, 50],  # Keep the focus along the Y-axis for a direct view of the road
        [0, 0, 0, 1]
    ])
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        # Update the visualizer with the new point cloud
        vis.poll_events()
        vis.update_renderer()
    
        # Add bounding boxes, render options, camera adjustments, and saving frames as before...
        output_image_path = os.path.join(output_folder, f"frame_{pcd_file.split('.')[0]}.jpg")
        vis.capture_screen_image(output_image_path)
        print(f"Saved {output_image_path}")
    
        # Delay between frames
        time.sleep(0.4)
    except Exception as e:
        print(e)

vis.destroy_window()
