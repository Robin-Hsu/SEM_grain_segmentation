# helper functions used in the main script
# Author: Chengchao Xu
# Date: 2025-03-11

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imsave
from skimage import feature, morphology, util
from scipy import ndimage as ndi
from skimage.feature import multiscale_basic_features, peak_local_max
from skimage.morphology import disk, erosion, dilation, closing, opening
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

# --------------- VISUALIZATION ----------------

def visualize_image(image, title=None, cmap='gray'):
    """
    Visualize an image.
    
    Args:
        image (ndarray): Input image
        title (str): Title for the plot
        cmap (str): Colormap
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap=cmap)
    
    if title is not None:
        plt.title(title)
    plt.show()

def visualize_labeled(labeled_image, random_colors=True):
    """
    Visualize a labeled image.
    
    Args:
        labeled_image (ndarray): Labeled image
        random_colors (bool): Whether to use random colors
    """
    plt.figure(figsize=(12, 10))
    
    if random_colors:
        # Create a random colormap
        rng = np.random.RandomState(0)
        n_labels = np.max(labeled_image) + 1
        colors = np.zeros((n_labels, 3))
        colors[0] = [0, 0, 0]  # Background is black
        colors[1:] = rng.rand(n_labels-1, 3)
        plt.imshow(labeled_image, cmap=plt.cm.colors.ListedColormap(colors))
    else:
        plt.imshow(labeled_image, cmap='nipy_spectral')
        
    plt.title("Labeled Image")
    plt.show()

def visualize_overlay(original_image, segmentation, alpha=0.3):
    """
    Visualize segmentation as an overlay on the original image.
    
    Args:
        original_image (ndarray): Original image
        segmentation (ndarray): Binary or labeled segmentation
        alpha (float): Transparency of the overlay
        
    Returns:
        ndarray: Overlay image for saving
    """
    # Create a color version of the original image if it's grayscale
    if len(original_image.shape) == 2:
        rgb_original = np.stack([original_image]*3, axis=-1)
    else:
        rgb_original = original_image.copy()
    
    # Normalize to 0-1 if needed
    rgb_original = rgb_original.astype(float) / np.max(rgb_original)
    
    # Create the overlay image
    overlay = rgb_original.copy()
    
    if segmentation.dtype == np.bool_ or (segmentation.max() == 1 and segmentation.min() == 0):
        # Binary overlay - use cool colormap (cyan/blue)
        mask = segmentation > 0
        overlay[mask] = overlay[mask] * (1-alpha) + np.array([0, 1, 1]) * alpha
    else:
        # Labeled overlay with random colors
        rng = np.random.RandomState(0)
        n_labels = np.max(segmentation) + 1
        
        # Generate random colors for each label
        colors = np.zeros((n_labels, 3))
        for i in range(1, n_labels):
            colors[i] = rng.rand(3)
        
        # Apply colors to the overlay
        for i in range(1, n_labels):
            mask = segmentation == i
            overlay[mask] = overlay[mask] * (1-alpha) + colors[i] * alpha
    
    # Display the overlay
    plt.figure(figsize=(12, 10))
    plt.imshow(overlay)
    plt.title("Segmentation Overlay")
    plt.axis('off')
    plt.show()
    
    # Convert to uint8 for saving (0-255 range)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    
    # Return the overlay image
    return overlay_uint8

def visualize_grain_statistics(grain_data):
    """
    Visualize statistics about the grains.
    
    Args:
        grain_data (dict): Dictionary of grain properties
    """
    # Print basic statistics
    print("\nGrain Analysis Results:")
    print(f"Number of grains: {grain_data['grain_count']}")
    print(f"Average grain area: {np.mean(grain_data['areas']):.2f} nm²")
    print(f"Average grain diameter: {np.mean(grain_data['diameters']):.2f} nm")
    print(f"25th percentile: {np.percentile(grain_data['diameters'], 25):.2f} nm")
    print(f"Median grain diameter: {np.median(grain_data['diameters']):.2f} nm")
    print(f"75th percentile: {np.percentile(grain_data['diameters'], 75):.2f} nm")

    # Set up the figure with a 1x4 grid
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    # 1. Histogram of grain diameters
    axs[0].hist(grain_data['diameters'], bins=30, edgecolor='black', color='skyblue')
    axs[0].set_xlabel('Grain Diameter (nm)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Grain Size Distribution')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Box plot of grain diameters
    bp = axs[1].boxplot(grain_data['diameters'], vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('skyblue')
    axs[1].set_ylabel('Grain Diameter (nm)')
    axs[1].set_title('Grain Size Quartiles')
    axs[1].set_xticks([])
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Histogram of grain areas
    axs[2].hist(grain_data['areas'], bins=30, edgecolor='black', color='lightgreen')
    axs[2].set_xlabel('Grain Area (nm²)')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Grain Area Distribution')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    # 4. Box plot of grain areas
    bp = axs[3].boxplot(grain_data['areas'], vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    axs[3].set_ylabel('Grain Area (nm²)')
    axs[3].set_title('Grain Area Quartiles')
    axs[3].set_xticks([])
    axs[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Add scale bar to the segmentation image
def add_simple_scale_bar(image, scale_factor=0.603, bar_length_nm=500):
    """Add a simple scale bar with text to an image"""
    
    img_with_bar = image.copy()
    height, width = img_with_bar.shape[0], img_with_bar.shape[1]
    
    # Calculate bar dimensions
    bar_length_px = int(bar_length_nm * scale_factor)
    bar_height = max(3, height // 100)
    margin = 20
    
    # Draw bar
    x_start = width - margin - bar_length_px
    x_end = width - margin
    y_pos = height - margin - bar_height
    
    # Place white scale bar
    if len(img_with_bar.shape) == 3:  # RGB image
        img_with_bar[y_pos:y_pos+bar_height, x_start:x_end] = [255, 255, 255]
        text_color = (255, 255, 255)
    else:  # Grayscale image
        img_with_bar[y_pos:y_pos+bar_height, x_start:x_end] = 255
        text_color = 255
    
    # Add text using OpenCV
    if len(img_with_bar.shape) == 2:
        img_with_bar = cv2.cvtColor(img_with_bar, cv2.COLOR_GRAY2BGR)
    
    # Text position
    text_x = x_start + bar_length_px // 2
    text_y = y_pos - 5
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, height / 1000)
    text = f"{bar_length_nm} nm"
    
    # Add black outline for visibility
    cv2.putText(img_with_bar, text, (text_x-1, text_y-1), font, font_scale, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_with_bar, text, (text_x+1, text_y-1), font, font_scale, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_with_bar, text, (text_x-1, text_y+1), font, font_scale, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_with_bar, text, (text_x+1, text_y+1), font, font_scale, (0,0,0), 2, cv2.LINE_AA)
    
    # Add white text
    cv2.putText(img_with_bar, text, (text_x, text_y), font, font_scale, (255,255,255), 1, cv2.LINE_AA)
    
    return img_with_bar


# --------------- MORPHOLOGICAL OPERATIONS ----------------
def connect_boundary_dots(image, line_length=5, num_angles=16, min_size=50):
    """
    Connect dotted lines at grain boundaries while suppressing noise spots
    
    Args:
        image: binary image with dotted outlines and noise
        line_length: length of line structuring element
        num_angles: number of angles to check for line connections
        min_size: minimum size of objects to keep
    
    Returns:
        processed image with connected boundaries
    """
    # 1. Enhance linear structures using multi-angle line elements
    enhanced = np.zeros_like(image, dtype=float)
    
    for angle in range(num_angles):
        # Create line structuring element at different angles
        theta = angle * np.pi / num_angles
        line_x = int(line_length * np.cos(theta))
        line_y = int(line_length * np.sin(theta))
        
        # Ensure the line has at least length 3
        if abs(line_x) < 2 and abs(line_y) < 2:
            if abs(line_x) >= abs(line_y):
                line_x = 3 * np.sign(line_x) if line_x != 0 else 3
                line_y = 0
            else:
                line_x = 0
                line_y = 3 * np.sign(line_y) if line_y != 0 else 3
        
        # Create structuring element (rectangle is used to create a line)
        if line_x == 0:
            selem = np.ones((abs(line_y), 1), dtype=bool)
        elif line_y == 0:
            selem = np.ones((1, abs(line_x)), dtype=bool)
        else:
            # For diagonal lines, need to create custom structuring element
            size = max(abs(line_x), abs(line_y))
            selem = np.zeros((size, size), dtype=bool)
            for i in range(size):
                x = int(i * line_x / size)
                y = int(i * line_y / size)
                if 0 <= x < size and 0 <= y < size:
                    selem[y, x] = True
        
        # Apply closing with this line element
        closed = closing(image, selem)
        enhanced = np.maximum(enhanced, closed)
    
    # 2. Remove internal noise spots using area opening
    cleaned = enhanced > 0.5  # Convert back to binary
    cleaned = morphology.remove_small_objects(cleaned, min_size=min_size)
    
    # 3. Fill small holes in boundaries to make them more robust
    filled = morphology.remove_small_holes(cleaned, area_threshold=min_size)
    
    # 4. Use edge enhancement filter to strengthen boundaries
    edges = feature.canny(filled.astype(float), sigma=1, low_threshold=0.1, high_threshold=0.2)
    result = np.logical_or(filled, edges)
    
    # 5. Clean up the final result
    final = opening(closing(result, disk(1)), disk(1))
    
    return final

def apply_erosion(image, radius=1, iterations=1):
    """
    Apply erosion to a binary image.
    
    Args:
        image (ndarray): Input binary image
        radius (int): Radius of the disk-shaped structuring element
        iterations (int): Number of times to apply the operation
        
    Returns:
        ndarray: Eroded image
    """
    selem = disk(radius)
    result = image.copy()
    
    for _ in range(iterations):
        result = erosion(result, selem)
        
    return result

def apply_dilation(image, radius=1, iterations=1):
    """
    Apply dilation to a binary image.
    
    Args:
        image (ndarray): Input binary image
        radius (int): Radius of the disk-shaped structuring element
        iterations (int): Number of times to apply the operation
        
    Returns:
        ndarray: Dilated image
    """
    selem = disk(radius)
    result = image.copy()
    
    for _ in range(iterations):
        result = dilation(result, selem)
        
    return result

def apply_opening(image, radius=1, iterations=1):
    """
    Morphological opening on an image is defined as an erosion followed by a dilation. 
    Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks.
    Args:
        image (ndarray): Input binary image
        radius (int): Radius of the disk-shaped structuring element
        iterations (int): Number of times to apply the operation
        
    Returns:
        ndarray: Opened image
    """
    selem = disk(radius)
    result = image.copy()
    
    for _ in range(iterations):
        result = opening(result, selem)
        
    return result

def apply_closing(image, radius=1, iterations=1):
    """
    Morphological closing on an image is defined as a dilation followed by an erosion. 
    Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks.
    Args:
        image (ndarray): Input binary image
        radius (int): Radius of the disk-shaped structuring element
        iterations (int): Number of times to apply the operation
        
    Returns:
        ndarray: Closed image
    """
    selem = disk(radius)
    result = image.copy()
    
    for _ in range(iterations):
        result = closing(result, selem)
        
    return result

# --------------- OBJECT MANIPULATION ----------------

def remove_small_objects(image, min_size=50):
    """
    Remove small objects from a binary image.
    
    Args:
        image (ndarray): Input binary image
        min_size (int): Minimum size of objects to keep
        
    Returns:
        ndarray: Filtered image
    """
    # Ensure we have a binary image
    binary = image > 0
    return morphology.remove_small_objects(binary, min_size=min_size)

def fill_grain_holes(binary_image, max_hole_size=100):
    """
    Fill holes in grains (black regions completely surrounded by white).
    
    Args:
        binary_image: Binary image where 1=grain, 0=background
        max_hole_size: Maximum size of holes to fill (pixels)
        
    Returns:
        Binary image with holes filled
    """
    # Make sure we have a boolean image
    binary = binary_image > 0
    
    # Find all holes in the image (black regions completely surrounded by white)
    # This creates a binary image where holes are 1 and everything else is 0
    holes = ndi.binary_fill_holes(binary) & ~binary
    
    # Label the holes
    labeled_holes = label(holes)
    
    # Create a copy of the input image
    filled_image = binary.copy()
    
    # Fill holes smaller than max_hole_size
    for region in regionprops(labeled_holes):
        if region.area <= max_hole_size:
            filled_image[labeled_holes == region.label] = True
    
    return filled_image

# --------------- OBJECT SEPARATION ----------------
def separate_watershed(image, min_distance=20, compactness=0):
    """
    Use watershed algorithm to separate touching objects.
    
    Args:
        image (ndarray): Input binary image
        min_distance (int): Minimum distance between local maxima
        compactness (float): Compactness parameter for the watershed algorithm
        
    Returns:
        tuple: Separated binary image, labeled image, distance map
    """
    # Ensure we have a binary image
    binary = image > 0
    
    # Create distance map
    distance = ndi.distance_transform_edt(binary)
    
    # Find local maxima as markers
    coords = peak_local_max(distance, min_distance=min_distance, exclude_border=False)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    
    # Apply watershed
    labels = watershed(-distance, markers, mask=binary, compactness=compactness)
        
    # Create binary result
    separated = np.zeros_like(labels, dtype=bool)
    separated[labels > 0] = True
    
    return separated, labels, distance

def separate_touching(segmented, min_distance, num_erosions=0):
    """
    Separates touching objects using a watershed process.
    
    Args:
        segmented (ndarray): pre-segmented binary image
        min_distance (int): minimum distance between separated objects
        num_erosions (int): number of erosions to perform at end of process
        
    Returns:
        tuple: separated binary image, labels, distance map
    """
    distance = ndi.distance_transform_edt(segmented)
    coords = peak_local_max(distance, min_distance=min_distance, exclude_border=False)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=segmented, watershed_line=True)
    separated = labels.copy()
    separated[labels > 0] = 1

    # Apply erosion if requested
    if num_erosions > 0:
        for i in range(num_erosions):
            separated = erosion(separated, disk(1))

    return separated, labels, distance

# --------------- ANALYSIS ----------------

def analyze_grains(labeled_image, scale_factor=1.0):
    """
    Analyze grain properties and convert to real units.
    
    Args:
        labeled_image (ndarray): Labeled image
        scale_factor (float): Scale factor in pixel/nm
        
    Returns:
        dict: Dictionary of grain properties
    """
    regions = regionprops(labeled_image)
    
    # Initialize arrays for measurements
    areas = []
    equivalent_diameters = []
    perimeters = []
    major_axis_lengths = []
    minor_axis_lengths = []
    eccentricities = []
    centroids = []
    
    for region in regions:
        if region.label == 0:  # Skip background
            continue
            
        # Convert measurements to nm
        area_nm = region.area / (scale_factor ** 2)
        areas.append(area_nm)
        
        diameter_nm = region.equivalent_diameter / scale_factor
        equivalent_diameters.append(diameter_nm)
        
        perimeter_nm = region.perimeter / scale_factor
        perimeters.append(perimeter_nm)
        
        major_axis_nm = region.major_axis_length / scale_factor
        major_axis_lengths.append(major_axis_nm)
        
        minor_axis_nm = region.minor_axis_length / scale_factor
        minor_axis_lengths.append(minor_axis_nm)
        
        eccentricities.append(region.eccentricity)
        centroids.append(region.centroid)
    
    return {
        'grain_count': len(areas),
        'areas': np.array(areas),
        'diameters': np.array(equivalent_diameters),
        'perimeters': np.array(perimeters),
        'major_axis_lengths': np.array(major_axis_lengths),
        'minor_axis_lengths': np.array(minor_axis_lengths),
        'eccentricities': np.array(eccentricities),
        'centroids': np.array(centroids),
        'aspect_ratios': np.array(major_axis_lengths) / np.array(minor_axis_lengths)
    }

def save_grain_analysis(grain_data, output_file="grain_analysis.csv"):
    """
    Save grain analysis data to CSV.
    
    Args:
        grain_data (dict): Dictionary of grain properties
        output_file (str): Output CSV file path
    """
    grain_df = pd.DataFrame({
        'Area_nm2': grain_data['areas'],
        'Diameter_nm': grain_data['diameters'],
        'Perimeter_nm': grain_data['perimeters'],
        'MajorAxis_nm': grain_data['major_axis_lengths'],
        'MinorAxis_nm': grain_data['minor_axis_lengths'],
        'AspectRatio': grain_data['aspect_ratios'],
        'Eccentricity': grain_data['eccentricities']
    })
    grain_df.to_csv(output_file, index=False)