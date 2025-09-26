#!/usr/bin/env python
"""
Demonstration script for using VCR1 JSON annotations with enhanced feature extraction.
This script shows how the enhanced feature extractor leverages pre-extracted annotations
from VCR1 JSON files to improve the feature extraction process.
"""

import os
import argparse
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import feature extractor
from dataloaders.enhanced_feature_extraction import EnhancedFeatureExtractor

def visualize_detections(image_path, json_path):
    """
    Visualize detections from a VCR JSON file
    
    Args:
        image_path: Path to the image
        json_path: Path to the VCR JSON file
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to RGB for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load JSON
    with open(json_path, 'r') as f:
        vcr_data = json.load(f)
    
    # Get boxes and names
    boxes = np.array(vcr_data['boxes'])
    names = vcr_data['names']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Draw boxes with labels
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, score = box
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            fill=False, edgecolor='red', linewidth=2
        ))
        plt.text(
            x1, y1, f"{names[i]}: {score:.2f}", 
            color='white', fontsize=10, 
            bbox=dict(facecolor='red', alpha=0.7)
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def compare_features(image_path, json_path):
    """
    Compare features with and without VCR JSON data
    
    Args:
        image_path: Path to the image
        json_path: Path to the VCR JSON file
    """
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor("temp_features")
    
    # Extract features without VCR data
    print("Extracting features without VCR JSON data...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    features_without_vcr = feature_extractor.extract_detectron2_features(image)
    
    # Extract features with VCR data
    print("Extracting features with VCR JSON data...")
    with open(json_path, 'r') as f:
        vcr_data = json.load(f)
    
    boxes = np.array(vcr_data['boxes'])[:,:4]
    names = vcr_data['names']
    segms = vcr_data['segms']
    
    # Map VCR class names to COCO class IDs where possible
    class_ids = []
    coco_name_to_id = {v: k for k, v in feature_extractor.coco_id_to_name.items()}
    
    for name in names:
        if name in coco_name_to_id:
            class_ids.append(coco_name_to_id[name])
        elif name == 'person':
            class_ids.append(0)  # person in COCO
        else:
            class_ids.append(-100)  # unknown class
    
    features_with_vcr = feature_extractor.extract_detectron2_features(
        image, boxes, class_ids, segms
    )
    
    # Print comparison
    print("\nComparison of features:")
    print(f"Without VCR data: {len(features_without_vcr['boxes'])} objects detected")
    print(f"With VCR data: {len(features_with_vcr['boxes'])} objects available")
    
    print("\nDetected classes without VCR data:")
    for i, name in enumerate(features_without_vcr['class_names']):
        print(f"  {i}: {name}")
    
    print("\nProvided classes with VCR data:")
    for i, name in enumerate(features_with_vcr['class_names']):
        print(f"  {i}: {name}")
    
    # Compare feature vectors
    print("\nFeature dimensions:")
    print(f"  Without VCR - Image features: {features_without_vcr['image_features'].shape}")
    print(f"  With VCR - Image features: {features_with_vcr['image_features'].shape}")
    
    return features_without_vcr, features_with_vcr

def process_vcr_directory(vcr_dir, output_dir):
    """
    Process all images in a VCR directory with their JSON annotations
    
    Args:
        vcr_dir: Directory containing VCR images and JSON files
        output_dir: Directory to save extracted features
    """
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor(output_dir)
    
    # Get all JSON files
    json_files = []
    for root, _, files in os.walk(vcr_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                json_files.append(json_path)
    
    print(f"Found {len(json_files)} JSON files in {vcr_dir}")
    
    # Process each image
    successful = 0
    for json_path in tqdm(json_files, desc="Processing VCR images"):
        # Get image path
        image_path = os.path.splitext(json_path)[0] + '.jpg'
        
        # Check if image exists
        if not os.path.exists(image_path):
            image_path = os.path.splitext(json_path)[0] + '.png'
            if not os.path.exists(image_path):
                print(f"Warning: No image found for {json_path}")
                continue
        
        # Process image with VCR JSON data
        if feature_extractor.process_image(image_path, json_path):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(json_files)} images")

def main():
    """Main function for the demonstration script"""
    parser = argparse.ArgumentParser(description="VCR JSON Feature Extraction Demo")
    parser.add_argument("--image_path", help="Path to an image file")
    parser.add_argument("--json_path", help="Path to a VCR JSON file")
    parser.add_argument("--vcr_dir", help="Directory containing VCR images and JSON files")
    parser.add_argument("--output_dir", default="vcr_features", 
                        help="Directory to save extracted features")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize detections from VCR JSON")
    parser.add_argument("--compare", action="store_true",
                        help="Compare features with and without VCR JSON data")
    
    args = parser.parse_args()
    
    if args.visualize and args.image_path and args.json_path:
        # Visualize detections
        visualize_detections(args.image_path, args.json_path)
        
    elif args.compare and args.image_path and args.json_path:
        # Compare features
        compare_features(args.image_path, args.json_path)
        
    elif args.vcr_dir:
        # Process all images in directory
        process_vcr_directory(args.vcr_dir, args.output_dir)
        
    else:
        print("Error: Please provide either:")
        print("  1. --image_path and --json_path with --visualize or --compare")
        print("  2. --vcr_dir to process all images in a directory")

if __name__ == "__main__":
    main()
