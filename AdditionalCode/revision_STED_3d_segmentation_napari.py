import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

from pathlib import Path
# import aicsimageio
from tifffile import TiffFile

from scipy.spatial.distance import cdist

from sklearn.manifold import TSNE

from tqdm import tqdm

# from read_files import read_cizlif_files_get_data, read_cizlif_file, from_files_to_df, read_files_pipeline


import gudhi as gd
from sklearn.preprocessing import MinMaxScaler
from gudhi.hera.wasserstein import wasserstein_distance
from gudhi.bottleneck import bottleneck_distance 
from gudhi.representations import Entropy, PersistenceImage, BettiCurve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from skimage.morphology import remove_small_objects, remove_small_holes, label
from skimage.morphology import binary_dilation
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from skimage.morphology import remove_small_holes


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as mpatches

import skimage
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing, disk, binary_closing, remove_small_holes
from skimage.measure import label, regionprops
from skimage.io import imsave
from itertools import product
from skimage.morphology import convex_hull_image

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

from skimage.morphology import convex_hull_image
import skimage
import skimage.io as skio

from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import data
from skimage import transform
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage import segmentation
from skimage.segmentation import expand_labels
from skimage import filters
from skimage import morphology
from skimage import measure

import numpy as np  # For array operations and numerical computations
from skimage import filters, morphology, segmentation, measure, draw  # For image processing and analysis
from scipy import ndimage as ndi  # For distance transform and other ndimage operations
from scipy.ndimage import binary_fill_holes
from skimage.draw import ellipse_perimeter
from skimage.morphology import binary_dilation, convex_hull_image, disk


from skimage.color import rgb2gray
from scipy.ndimage import label
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.preprocessing import LabelEncoder

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('font',**{'family':'serif','serif':['Arial']})
rc('text', usetex=True)

# import cv2 as cv

import napari

def strutcureDiagonal(ndim):
    if ndim == 1:
        return np.ones([3])
    elif ndim == 2:
        return np.ones([3, 3])
    elif ndim == 3:
        return np.ones([3, 3, 3])
    else:
        raise ValueError('ndim must be 1, 2 or 3')

def watershed_segmentation_2d(image, only_largest=True, threshold_dist=True, verbatim=False):
    """
    Perform watershed segmentation on a 2D image using pixels with values
    above the 90%-quantile as markers.
    
    Parameters:
        image (numpy.ndarray): 2D array representing the image to segment.
        
    Returns:
        segmented (numpy.ndarray): Segmented image with unique labels for each region.
    """
    # Compute the edges of the image for watershed algorithm
    edges = filters.scharr(image)
    
    # Denoise the image using a median filter
    denoised = ndi.median_filter(image, size=3)
    
    # Threshold the denoised image using Li's method
    threshold = filters.threshold_li(denoised)
    binary_image = denoised > threshold
    
    # Clean up small holes and objects in the binary image
    cleaned_image = morphology.remove_small_holes(binary_image, area_threshold=20**2)
    cleaned_image = morphology.remove_small_objects(cleaned_image, min_size=20**2)
    
    # Compute distance transform for marker identification
    distance = ndi.distance_transform_edt(cleaned_image)
    
    # Identify markers as local maxima above the 90%-quantile of the distance map
    if threshold_dist:
        quantile_threshold = np.percentile(distance, 90)
        markers = (distance > quantile_threshold).astype(np.uint32)
        markers, _ = label(markers)
    else:
        quantile_threshold = np.percentile(image, 90)
        markers = (image > quantile_threshold).astype(np.uint32)
        markers, _ = label(markers)
    
    # Perform watershed segmentation
    segmented = segmentation.watershed(edges, markers, mask=cleaned_image)

    # Join all components which are adjoined and where segmented > 0
    labelled, _ = label(segmented > 0, structure=strutcureDiagonal(segmented.ndim))

    if only_largest:
        # Find the largest component
        largest_component = (labelled == np.argmax(np.bincount(labelled.ravel())[1:]) + 1)

        # Thicken the largest component using morphological dilation
        thickened_component = binary_dilation(largest_component, footprint=disk(5))  # Adjust disk size as needed

        # Compute the convex hull of the thickened component
        convex_hull = convex_hull_image(thickened_component)

        # Update segmented to include the convex hull
        segmented = convex_hull.astype(np.uint32)

        return segmented

    else:
        num_cc = 0
        iteration = 0
        while num_cc != np.max(labelled):
            num_cc = np.max(labelled)
            segmented_new = np.zeros_like(labelled)
            for i in np.unique(labelled):
                if i == 0:
                    continue
                component = (labelled == i)
                convex_hull = convex_hull_image(component)
                segmented_new[convex_hull != 0] = i

            labelled, _ = label(segmented_new > 0, structure=strutcureDiagonal(segmented.ndim))
            iteration += 1
            if iteration > 1 and verbatim:
                print(f'CCs joined in watershed_segmentation_2d, step {iteration}')
            if iteration > 10:
                print('Warning: Iteration > 10 in watershed_segmentation_2d')
                break
        return segmented_new
    

files_origin = Path('NewData_2024', 'data', 'images', 'STED_images_for_revision')
metadata = pd.read_csv(Path(files_origin, 'metadata_sted_extract_images_with_zslices.csv'))
images = np.load(Path(files_origin, 'images_sted_extract_images_with_zslices.npz'))
# images are of the form 'image_xxxx' and labels are given in labels where ES is 0 and NS is 1
labels = images['labels']
images = [images[k].astype(np.float32) for k in sorted([x for x in images.keys() if 'labels' not in x])]


# import numpy as np
# import napari
# from skimage.filters import sobel
# from skimage.measure import label, regionprops
# from skimage.segmentation import expand_labels, watershed
# import os
# from magicgui import magic_factory
# from qtpy.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget
# import signal
# import sys

# # Create output directories if they don't exist
# os.makedirs('mask_outputs', exist_ok=True)
# os.makedirs('Temporary', exist_ok=True)  # Added Temporary folder as requested

# # Add keyboard interrupt handler
# def signal_handler(sig, frame):
#     print('Process interrupted by user. Exiting...')
#     sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)

# # Process each image in the dataset
# for img_count, img in enumerate(images):
#     print(f"Processing image {img_count+1}/{len(images)}")
    
#     # Create a napari viewer for the interactive segmentation
#     viewer = napari.Viewer(title=f"Interactive Segmentation - Image {img_count+1}")
    
#     # Add the original image (max projection)
#     img_agg = np.max(img, axis=0)
#     image_layer = viewer.add_image(img_agg, name='Max Projection')
    
#     # Add the original 3D image as well
#     # Add the original 3D image **last** to ensure it's the highest layer
#     volume_layer = viewer.add_image(
#         img, 
#         name='3D Volume', 
#         visible=True, 
#         contrast_limits=(np.min(img), np.max(img)), 
#         colormap='gray',
#         opacity=1
#     )

#     # Move "3D Volume" to the top of the layer stack
#     viewer.layers.move(viewer.layers.index("3D Volume"), len(viewer.layers) - 1)
    
#     # Store current parameters in a dictionary to avoid nonlocal issues
#     params = {
#         'bg_threshold': 1.0,
#         'fg_threshold': 21.0,
#         'min_area': 150,
#         'expand_distance': 50
#     }
    
#     # Function to update segmentation based on current thresholds
#     def update_segmentation():
#         # Create markers using current thresholds
#         markers = np.zeros_like(img_agg, dtype=np.int32)
#         markers[img_agg < params['bg_threshold']] = 2  # Background
#         markers[img_agg > params['fg_threshold']] = 1  # Foreground
        
#         # Update markers layer
#         if 'Markers' in viewer.layers:
#             viewer.layers['Markers'].data = markers
#         else:
#             # Add markers layer - fixed to use label_colors instead of color
#             markers_layer = viewer.add_labels(markers, name='Markers')
#             # Set custom colors for the markers
#             markers_layer.color = {
#                 0: [0, 0, 0, 0],  # Transparent for unassigned
#                 1: [0, 1, 0, 0.5],  # Green for foreground
#                 2: [1, 0, 0, 0.5]   # Red for background
#             }
        
#         # Calculate edges
#         edges = sobel(img_agg)
        
#         # Update edges layer
#         if 'Edges' in viewer.layers:
#             viewer.layers['Edges'].data = edges
#         else:
#             viewer.add_image(edges, name='Edges', colormap='cividis', visible=True)
        
#         # Apply watershed segmentation
#         ws = watershed(edges, markers)
#         seg1 = label(ws == 1)  # foreground
        
#         # Remove small regions
#         for region in regionprops(seg1):
#             if region.area < params['min_area']:
#                 seg1[seg1 == region.label] = 0
        
#         # Expand labels to fill gaps
#         expanded = expand_labels(seg1, distance=params['expand_distance'])
        
#         # Relabel for consistency
#         unique_labels = np.unique(expanded)
#         unique_labels = unique_labels[unique_labels != 0]  # Exclude background
#         new_expanded = np.zeros_like(expanded)
#         for i, vali in enumerate(unique_labels):
#             new_expanded[expanded == vali] = i + 1
#         expanded = new_expanded
        
#         # Remove segments touching the image boundary
#         border_labels = set()
#         for border in [expanded[:,0], expanded[:,-1], expanded[0,:], expanded[-1,:]]:
#             border_labels.update(np.unique(border))
        
#         for vali in border_labels:
#             if vali != 0:  # Don't remove background
#                 expanded[expanded == vali] = 0
        
#         # Update segmentation layer
#         if '2D Segmentation' in viewer.layers:
#             viewer.layers['2D Segmentation'].data = expanded
#         else:
#             viewer.add_labels(expanded, name='2D Segmentation')
            
#         # Create 3D segmentation from 2D expanded labels
#         expanded_3d = np.zeros_like(img, dtype=np.int32)
        
#         # For each unique label in the 2D segmentation
#         for label_id in np.unique(expanded)[1:]:  # Skip background (0)
#             # Create a binary mask for this label
#             label_mask_2d = expanded == label_id
            
#             # For each z-slice
#             for z in range(img.shape[0]):
#                 # Only include pixels where the intensity is above a threshold
#                 # and is within the 2D segmented region
#                 z_slice = img[z]
#                 # Use a relative threshold (30% of max intensity in this region)
#                 thresh = 0.3 * np.max(z_slice[label_mask_2d]) if np.any(label_mask_2d) else 0
#                 # Assign the label where intensity is sufficient
#                 expanded_3d[z][np.logical_and(label_mask_2d, z_slice > thresh)] = label_id
                
#         # Update 3D segmentation layer
#         if '3D Segmentation' in viewer.layers:
#             viewer.layers['3D Segmentation'].data = expanded_3d
#         else:
#             viewer.add_labels(expanded_3d, name='3D Segmentation', visible=True)
            
#         return expanded_3d
    
#     # Create control panel
#     @magic_factory(
#         bg_threshold={'widget_type': 'FloatSlider', 'min': 0.0, 'max': 10.0, 'step': 0.1},
#         fg_threshold={'widget_type': 'FloatSlider', 'min': 5.0, 'max': 50.0, 'step': 0.5},
#         min_area={'widget_type': 'Slider', 'min': 10, 'max': 500, 'step': 10},
#         expand_distance={'widget_type': 'Slider', 'min': 1, 'max': 100, 'step': 1}
#     )
#     def segmentation_controls(
#         bg_threshold=1.0, 
#         fg_threshold=21.0, 
#         min_area=150, 
#         expand_distance=50, 
#         apply_button=True, 
#         show_markers=False,
#         show_edges=False,
#         show_2d_seg=True,
#         show_3d_seg=True,
#         show_volume=True
#     ):
#         # Update parameter values in our dictionary
#         params['bg_threshold'] = bg_threshold
#         params['fg_threshold'] = fg_threshold
#         params['min_area'] = min_area
#         params['expand_distance'] = expand_distance
        
#         # Toggle layer visibility
#         if 'Markers' in viewer.layers:
#             viewer.layers['Markers'].visible = show_markers
#         if 'Edges' in viewer.layers:
#             viewer.layers['Edges'].visible = show_edges
#         if '2D Segmentation' in viewer.layers:
#             viewer.layers['2D Segmentation'].visible = show_2d_seg
#         if '3D Segmentation' in viewer.layers:
#             viewer.layers['3D Segmentation'].visible = show_3d_seg
#         if '3D Volume' in viewer.layers:
#             viewer.layers['3D Volume'].visible = show_volume
        
            
#         # Update segmentation
#         if apply_button:
#             return update_segmentation()
    
#     # Add controls to the viewer
#     controls = segmentation_controls()
#     viewer.window.add_dock_widget(controls, area='right')
    
#     # Do initial segmentation
#     expanded_3d = update_segmentation()
    
#     # Define callback for interactive cleaning
#     @viewer.mouse_drag_callbacks.append
#     def _on_click(viewer, event):
#         if event.button == 1 and event.type == 'mouse_press':
#             # Get the current active layer
#             active_layer = viewer.layers.selection.active
            
#             # Check if we're working with a labels layer
#             if active_layer is not None and isinstance(active_layer, napari.layers.Labels):
#                 # Convert data coordinates to layer coordinates
#                 layer_coords = active_layer.world_to_data(event.position)
#                 layer_coords = tuple(map(int, layer_coords))
                
#                 # Make sure we're within bounds
#                 shape = active_layer.data.shape
#                 if all(0 <= c < s for c, s in zip(layer_coords, shape)):
#                     # Get the label value at the clicked position
#                     label_value = active_layer.data[layer_coords]
                    
#                     # Skip if background is clicked
#                     if label_value == 0:
#                         return
                        
#                     # Set the label value to 0 (deselect) in all dimensions
#                     active_layer.data[active_layer.data == label_value] = 0
                    
#                     print(f"Removed label {label_value} from {active_layer.name}")

#     # Function to save segmentation results
#     def save_segmentation(final_labels, directory):
#         # Get the unique labels
#         unique_labels = np.unique(final_labels)
#         unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        
#         # Save each mask as a separate file
#         for label_id in unique_labels:
#             # Create binary mask for this label
#             mask_3d = final_labels == label_id
            
#             # Get bounding box for this mask
#             z_indices, y_indices, x_indices = np.where(mask_3d)
#             if len(z_indices) == 0:  # Skip empty masks
#                 continue
                
#             z_min, z_max = np.min(z_indices), np.max(z_indices)
#             y_min, y_max = np.min(y_indices), np.max(y_indices)
#             x_min, x_max = np.min(x_indices), np.max(x_indices)
            
#             # Extract the bounding box region
#             bbox = {
#                 'mask': mask_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1],
#                 'bbox_coords': [z_min, z_max, y_min, y_max, x_min, x_max],
#                 'original_label': label_id
#             }
            
#             # Save the mask
#             mask_filename = f"{directory}/masks_{img_count}_{label_id}.npy"
#             np.save(mask_filename, bbox)
#             print(f"Saved mask to {mask_filename}")
        
#         return len(unique_labels)
    
#     # Add a button to finalize and save
#     def on_save_button_clicked():
#         # Get the 3D segmentation layer
#         if '3D Segmentation' in viewer.layers:
#             final_labels = viewer.layers['3D Segmentation'].data
            
#             # Save the masks to mask_outputs directory
#             count = save_segmentation(final_labels, 'mask_outputs')
#             print(f"Saved {count} masks for image {img_count}")
            
#             # Close the viewer
#             viewer.close()
#         else:
#             print("No 3D segmentation to save")
    
#     # Create a save button widget
#     save_button = QPushButton("Save Masks and Continue to Next Image")
#     save_button.clicked.connect(on_save_button_clicked)
    
#     # Add the save button to the viewer
#     save_widget = QWidget()
#     layout = QVBoxLayout()
#     layout.addWidget(QLabel("When finished with manual corrections:"))
#     layout.addWidget(save_button)
#     save_widget.setLayout(layout)
#     viewer.window.add_dock_widget(save_widget, area='bottom')

#     # **Close & Stop Button to Exit the Loop**
#     def on_stop_button_clicked():
#         global stop_flag
#         stop_flag = True  # Set flag to break loop
#         viewer.close()
#         print("Loop stopped by user.")

#     stop_button = QPushButton("Close & Stop Loop")
#     stop_button.clicked.connect(on_stop_button_clicked)

#     stop_widget = QWidget()
#     stop_layout = QVBoxLayout()
#     stop_layout.addWidget(QLabel("Click to close and stop loop:"))
#     stop_layout.addWidget(stop_button)
#     stop_widget.setLayout(stop_layout)
#     viewer.window.add_dock_widget(stop_widget, area='bottom')
    
#     # Define window close event to save to Temporary folder
#     def on_viewer_closing(event=None):
#         if '3D Segmentation' in viewer.layers:
#             final_labels = viewer.layers['3D Segmentation'].data
#             # Save to Temporary folder on window close
#             count = save_segmentation(final_labels, 'Temporary')
#             print(f"Window closing: Auto-saved {count} masks to Temporary folder for image {img_count}")
    
#     # Connect the close event
#     from qtpy.QtWidgets import QApplication

#     # Handle window close event
#     def on_viewer_closing(event=None):
#         if '3D Segmentation' in viewer.layers:
#             final_labels = viewer.layers['3D Segmentation'].data
#             save_segmentation(final_labels, 'Temporary')
#             print(f"Auto-saved masks to Temporary for image {img_count}")
        
#         # Ensure Qt properly exits
#         app = QApplication.instance()
#         if app is not None:
#             app.quit()

#     # Attach close event
#     viewer.window._qt_window.closeEvent = on_viewer_closing
    
#     # Add help text overlay
#     viewer.text_overlay.visible = True
#     viewer.text_overlay.text = ("Left-click on segments to remove them.\n"
#                                "Adjust parameters in control panel to refine segmentation.\n"
#                                "Click 'Save Masks and Continue' when finished.\n"
#                                "Press Ctrl+C to interrupt the process.")
    
#     # Start the napari event loop and wait for it to close
#     napari.run()

#     import gc

#     # Explicitly delete viewer and trigger garbage collection
#     del viewer
#     gc.collect()
    
#     # Check if the viewer window is still open (should not happen with napari.run())
#     try:
#         if hasattr(viewer, 'window') and hasattr(viewer.window, '_qt_window') and viewer.window._qt_window.isVisible():
#             print("Waiting for window to close before continuing...")
#             # Wait for the window to close
#             while viewer.window._qt_window.isVisible():
#                 # Pause briefly to avoid high CPU usage
#                 import time
#                 time.sleep(0.1)
#     except Exception as e:
#         print(f"Error checking window state: {e}")
    
#     print(f"Window closed for image {img_count}. Moving to next image.")
    
# print("Processing complete!")

import argparse

parser = argparse.ArgumentParser(description="Run Napari segmentation script.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing segmentations (default: False)")
args = parser.parse_args()

overwrite = args.overwrite

import numpy as np
import napari
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels, watershed
import os
from magicgui import magic_factory
from qtpy.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget
import signal
import sys
import glob
import matplotlib.pyplot as plt

def mask_exists(index):
    """Checks if any masks exist for the given index."""
    mask_files = glob.glob(f"mask_outputs/masks_{index}_*.npz")  # Check for .npz files
    return len(mask_files) > 0

def save_segmentation_plot(img_agg, mask_2d, mask_3d, filename):
    """Saves a matplotlib figure showing the original image, 2D mask, and 3D segmentation projection."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # **First subplot: Original image**

    axes[0].imshow(img_agg, cmap='gray')
    axes[0].set_title("Original Image")

    # **Second subplot: 2D Mask**
    axes[1].imshow(mask_2d, cmap='viridis', alpha=0.7)
    axes[1].set_title("2D Segmentation")

    # **Third subplot: 3D Projection**
    mask_3d_projection = np.max(mask_3d, axis=0)  # Collapse along Z
    axes[2].imshow(img_agg, cmap='gray', vmin=np.min(img_agg), vmax=np.quantile(img_agg, 0.95))
    axes[2].imshow(mask_3d_projection, cmap='magma', alpha=0.7)
    axes[2].set_title("3D Segmentation Projection")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)  # Close to avoid memory leaks
    print(f"Saved segmentation plot to {filename}")

# Create output directories if they don't exist
os.makedirs('mask_outputs', exist_ok=True)
os.makedirs('Temporary', exist_ok=True)

# Keyboard interrupt handler
def signal_handler(sig, frame):
    print('Process interrupted by user. Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Global index to track the current image
current_index = 0
stop_flag = False  # Flag to stop the loop

# Create Napari viewer **only once**
viewer = napari.Viewer(title="Interactive Segmentation")

# **Function to load and update images dynamically**
def load_image(img_index):
    """Load a new image into Napari without closing the viewer. Skips already processed images if overwrite=False."""
    global current_index
    current_index = img_index

    # **Skip existing masks if overwrite=False**
    while not overwrite and mask_exists(current_index):
        print(f"Skipping image {current_index} (mask already exists)")
        current_index += 1  # Move to the next image

        # **Stop when all images are processed**
        if current_index >= len(images):
            print("All images processed!")
            viewer.text_overlay.text = "All images processed!"
            return

    print(f"Processing image {current_index+1}/{len(images)}")

    img = images[current_index]
    img_agg = np.max(img, axis=0)

    # Clear old layers before adding new ones
    viewer.layers.clear()

    # Add the new image layers
    viewer.add_image(img_agg, name='Max Projection', visible=False)
    viewer.add_image(
        img,
        name='3D Volume',
        visible=True,
        contrast_limits=(np.min(img), np.quantile(img, 0.95)),
        colormap='gray',
        opacity=1
    )

    # Move "3D Volume" to the top of the layer stack
    viewer.layers.move(viewer.layers.index("3D Volume"), len(viewer.layers) - 1)
    # **Ensure segment info updates**
    update_segmentation()

# **Segmentation Logic**
params_base = {'bg_threshold': 3.0, 'fg_threshold': 21.0, 'min_area': 150, 'expand_distance': 50}
params = {'bg_threshold': 3.0, 'fg_threshold': 21.0, 'min_area': 150, 'expand_distance': 50}

# Function to remove a clicked segment
def remove_segment(viewer, event):
    """Removes a clicked segmentation only from the '2D Segmentation' layer and updates both 2D and 3D segmentation."""
    if event.button == 1 and event.type == 'mouse_press':  # Left click
        # **Ensure '2D Segmentation' is the active layer**
        if '2D Segmentation' in viewer.layers:
            viewer.layers.selection.active = viewer.layers['2D Segmentation']
        
        active_layer = viewer.layers.selection.active
        if active_layer and active_layer.name == "2D Segmentation":  # Ensure we're in 2D Segmentation
            coords = tuple(map(int, active_layer.world_to_data(event.position)))
            if all(0 <= c < s for c, s in zip(coords, active_layer.data.shape)):  # Ensure valid coordinates
                label_value = active_layer.data[coords]
                if label_value != 0:
                    # **Remove the segment from the 2D segmentation**
                    modified_2d_segmentation = active_layer.data.copy()
                    modified_2d_segmentation[modified_2d_segmentation == label_value] = 0
                    active_layer.data = modified_2d_segmentation  # **Reassign the .data array**
                    active_layer.refresh()
                    print(f"Removed segment {label_value} from 2D Segmentation")

                    # **Now update 3D segmentation to reflect the removal**
                    if '3D Segmentation' in viewer.layers:
                        three_d_layer = viewer.layers['3D Segmentation']
                        modified_3d_segmentation = three_d_layer.data.copy()
                        modified_3d_segmentation[modified_3d_segmentation == label_value] = 0
                        three_d_layer.data = modified_3d_segmentation  # **Reassign the .data array**
                        three_d_layer.refresh()
                        print(f"Also removed segment {label_value} from 3D Segmentation")

                    # **Manually refresh viewer layers to force visual update**
                    viewer.layers['2D Segmentation'].refresh()
                    viewer.layers['3D Segmentation'].refresh()
                    # **Ensure segment info updates**
                    update_segment_info()

    yield  # Required for Napari event handling

# Attach the on-click event (ensure it's not duplicated)
if remove_segment not in viewer.mouse_drag_callbacks:
    viewer.mouse_drag_callbacks.append(remove_segment)

def update_segmentation():
    """Perform segmentation and update layers."""
    img = images[current_index]
    img_agg = np.max(img, axis=0)

    # Create markers using thresholds
    markers = np.zeros_like(img_agg, dtype=np.int32)
    markers[img_agg < params['bg_threshold']] = 2
    markers[img_agg > params['fg_threshold']] = 1

    # Add/update markers layer
    if 'Markers' in viewer.layers:
        viewer.layers['Markers'].data = markers
    else:
        markers_layer = viewer.add_labels(markers, name='Markers', visible=False)

    # Compute edges
    edges = sobel(img_agg)

    # Add/update edges layer
    if 'Edges' in viewer.layers:
        viewer.layers['Edges'].data = edges
    else:
        viewer.add_image(edges, name='Edges', colormap='cividis', visible=False)

    # Apply watershed segmentation
    ws = watershed(edges, markers)
    seg1 = label(ws == 1)

    # Remove small regions
    for region in regionprops(seg1):
        if region.area < params['min_area']:
            seg1[seg1 == region.label] = 0

    # Expand labels
    expanded = expand_labels(seg1, distance=params['expand_distance'])

    # Remove boundary labels
    border_labels = set(np.unique(expanded[:, 0])) | set(np.unique(expanded[:, -1])) | \
                    set(np.unique(expanded[0, :])) | set(np.unique(expanded[-1, :]))

    for label_id in border_labels:
        if label_id != 0:
            expanded[expanded == label_id] = 0

    # Add/update 2D segmentation
    if '2D Segmentation' in viewer.layers:
        viewer.layers['2D Segmentation'].data = expanded
    else:
        viewer.add_labels(expanded, name='2D Segmentation', visible=False, opacity=0.6)

    # Generate 3D segmentation
    # expanded_3d = np.zeros_like(img, dtype=np.int32)
    # for label_id in np.unique(expanded)[1:]:
    #     label_mask_2d = expanded == label_id
    #     for z in range(img.shape[0]):
    #         z_slice = img[z]
    #         # thresh = 0.3 * np.max(z_slice[label_mask_2d]) if np.any(label_mask_2d) else 0
    #         expanded_3d[z][np.logical_and(label_mask_2d, z_slice > thresh)] = label_id
    ## TODO: make a better version
    expanded_3d = np.zeros_like(img, dtype=np.int32)
    for zi in range(img.shape[0]):
        expanded_3d[zi] = expanded

    # Add/update 3D segmentation
    if '3D Segmentation' in viewer.layers:
        viewer.layers['3D Segmentation'].data = expanded_3d
    else:
        viewer.add_labels(expanded_3d, name='3D Segmentation', visible=True, opacity=0.6)
    
    # **Ensure segment info updates**
    update_segment_info()

    return expanded_3d

# **Function to Save Results**
# def save_segmentation():
#     """Save the segmentation results."""
#     if '3D Segmentation' in viewer.layers:
#         final_labels = viewer.layers['3D Segmentation'].data
#         unique_labels = np.unique(final_labels)
#         unique_labels = unique_labels[unique_labels != 0]

#         for label_id, label_val in enumerate(unique_labels):
#             mask_3d = final_labels == label_val
#             mask_filename = f"mask_outputs/masks_{current_index}_{label_id}.npy"
#             np.save(mask_filename, mask_3d)
#             print(f"Saved mask to {mask_filename}")
def save_segmentation(final_labels, directory, img, img_agg, current_index):
    """Saves the segmentation bounding box, mask, and values as an .npz file.
       Also saves a visualization plot of the segmentation."""
    
    unique_labels = np.unique(final_labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    
    for label_idx, label_id in enumerate(unique_labels):
        mask_3d = final_labels == label_id  # Binary mask for this label
        
        # **Get bounding box for this mask**
        z_indices, y_indices, x_indices = np.where(mask_3d)
        if len(z_indices) == 0:  # Skip empty masks
            continue
        
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        # **Extract the bounding box region**
        mask_3d_cropped = mask_3d[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

        # **Extract 2D mask from max projection**
        mask_2d = np.max(mask_3d, axis=0)  # Collapse along Z-axis for 2D mask

        # **Extract values inside the mask**
        values = np.full_like(mask_3d_cropped, np.nan, dtype=img.dtype)
        if np.issubdtype(img.dtype, np.floating):
            values[mask_3d_cropped] = img[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1][mask_3d_cropped]
        else:
            values[mask_3d_cropped] = img[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1][mask_3d_cropped]
            values[~mask_3d_cropped] = np.min(img) - 1  # Set outside values to min-1
        
        # **Save as .npz**
        mask_filename = f"{directory}/masks_{current_index}_{label_idx}.npz"
        np.savez(mask_filename, 
                 bounding_box=np.array([z_min, z_max, y_min, y_max, x_min, x_max]), 
                 mask_2d=mask_2d, 
                 mask_3d=mask_3d_cropped, 
                 values=values)
        print(f"Saved mask to {mask_filename}")

        # **Save a visualization plot**
        save_segmentation_plot(img_agg, mask_2d, mask_3d, mask_filename.replace('.npz', '.png'))

# **Function to go to the next image**
# def next_image():
#     """Save segmentation and load the next image."""
#     global stop_flag

#     if stop_flag:
#         print("Loop stopped by user.")
#         return

#     save_segmentation()

#     if current_index + 1 < len(images):
#         load_image(current_index + 1)
#         update_segmentation()
#     else:
#         print("All images processed!")
#         viewer.text_overlay.text = "All images processed!"
#         viewer.text_overlay.visible = True
def next_image():
    """Save segmentation and load the next image, skipping processed ones."""
    global stop_flag

    if stop_flag:
        print("Loop stopped by user.")
        return

    # Save segmentation
    if '3D Segmentation' in viewer.layers:
        final_labels = viewer.layers['3D Segmentation'].data
        save_segmentation(final_labels, "mask_outputs", images[current_index], np.max(images[current_index], axis=0), current_index)

    global params
    for key in params.keys():
        params[key] = params_base[key]
    print('set to base params', params)

    # **Update UI sliders**
    segmentation_controls_ui.bg_threshold.value = params['bg_threshold']
    segmentation_controls_ui.fg_threshold.value = params['fg_threshold']
    segmentation_controls_ui.min_area.value = params['min_area']
    segmentation_controls_ui.expand_distance.value = params['expand_distance']

    # Move to the next index, skipping processed images
    load_image(current_index + 1)

    # Update segmentation only if a new image is loaded
    if current_index < len(images):
        update_segmentation()

# **Function to stop the loop and exit**
def stop_loop():
    """Stops the loop and exits Napari."""
    global stop_flag
    stop_flag = True
    print("Loop manually stopped.")
    viewer.close()

def next_image_without_saving():
    """Move to the next image without saving segmentation."""
    global stop_flag
    global params
    
    # **Reset segmentation parameters**
    for key in params.keys():
        params[key] = params_base[key]
    print('Segmentation parameters reset to base:', params)

    # **Update UI sliders**
    segmentation_controls_ui.bg_threshold.value = params['bg_threshold']
    segmentation_controls_ui.fg_threshold.value = params['fg_threshold']
    segmentation_controls_ui.min_area.value = params['min_area']
    segmentation_controls_ui.expand_distance.value = params['expand_distance']

    if stop_flag:
        print("Loop stopped by user.")
        return

    # Move to the next index, skipping processed images
    load_image(current_index + 1)

    # Update segmentation only if a new image is loaded
    if current_index < len(images):
        update_segmentation()

# **Add UI Buttons**
# Save & Next button
save_button = QPushButton("Save & Next Image")
save_button.clicked.connect(next_image)

save_widget = QWidget()
layout = QVBoxLayout()
layout.addWidget(QLabel("Click to save and move to the next image:"))
layout.addWidget(save_button)
save_widget.setLayout(layout)
viewer.window.add_dock_widget(save_widget, area='bottom')

# **Next Image Without Saving Button**
next_no_save_button = QPushButton("Next Image Without Saving")
next_no_save_button.clicked.connect(next_image_without_saving)

next_no_save_widget = QWidget()
next_no_save_layout = QVBoxLayout()
next_no_save_layout.addWidget(QLabel("Move to next image without saving:"))
next_no_save_layout.addWidget(next_no_save_button)
next_no_save_widget.setLayout(next_no_save_layout)

viewer.window.add_dock_widget(next_no_save_widget, area='bottom')

# Stop & Close button
stop_button = QPushButton("Stop & Exit")
stop_button.clicked.connect(stop_loop)

stop_widget = QWidget()
stop_layout = QVBoxLayout()
stop_layout.addWidget(QLabel("Click to stop and exit:"))
stop_layout.addWidget(stop_button)
stop_widget.setLayout(stop_layout)
viewer.window.add_dock_widget(stop_widget, area='bottom')



from qtpy.QtWidgets import QVBoxLayout, QLabel

# **Create Segment Info Widget**
segment_widget = QWidget()
segment_layout = QVBoxLayout()

segment_label = QLabel("Segments: 0")  # Placeholder text
segment_layout.addWidget(QLabel("Current Segments:"))
segment_layout.addWidget(segment_label)

segment_widget.setLayout(segment_layout)

# **Add to Napari Dock**
viewer.window.add_dock_widget(segment_widget, area="left")

# **Function to Update the Segment Display**
def update_segment_info():
    """Updates the UI with the current segment IDs and count."""
    if '2D Segmentation' in viewer.layers:
        segment_ids = np.unique(viewer.layers['2D Segmentation'].data)
        segment_ids = segment_ids[segment_ids != 0]  # Exclude background (0)
        
        segment_text = f"Segments: {len(segment_ids)}\n" + ", ".join(map(str, segment_ids))
    else:
        segment_text = "No segmentation available."

    segment_label.setText(segment_text)  # Update UI widget

def get_segment_size_range():
    """Returns the min and max segment size based on current segmentation."""
    segmentation_result = update_segmentation()  # Ensure segmentation is up to date
    segment_sizes = [region.area for region in regionprops(label(segmentation_result))]
    min_seg_size = min(segment_sizes) if segment_sizes else 10
    max_seg_size = max(segment_sizes) if segment_sizes else 500
    return min_seg_size, max_seg_size

# Fetch dynamic min/max values
min_seg_size, max_seg_size = get_segment_size_range()

@magic_factory(
    bg_threshold={'widget_type': 'FloatSlider', 'min': 0.0, 'max': 10.0, 'step': 0.1},
    fg_threshold={'widget_type': 'FloatSlider', 'min': 5.0, 'max': 50.0, 'step': 0.5},
    min_area={'widget_type': 'Slider', 'min': 150, 'max': 2000, 'step': 50},
    expand_distance={'widget_type': 'Slider', 'min': 1, 'max': 100, 'step': 1}
)
def segmentation_controls(
    bg_threshold=3.0, 
    fg_threshold=21.0, 
    min_area=150, 
    expand_distance=50
):
    """Control panel for segmentation adjustments."""
    params['bg_threshold'] = bg_threshold
    params['fg_threshold'] = fg_threshold
    params['min_area'] = min_area
    params['expand_distance'] = expand_distance
    update_segmentation()
    update_segment_info()  # Update segment display

# Create the segmentation controls widget
segmentation_controls_ui = segmentation_controls()
viewer.window.add_dock_widget(segmentation_controls_ui, area='right')

# Load the first image
load_image(0)

# Run Napari (keeps the viewer open)
napari.run()