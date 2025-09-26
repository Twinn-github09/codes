"""
Enhanced Feature Extraction for Visual Backstory Generation.
This module provides improved feature extraction using state-of-the-art models
to capture detailed visual information for generating backstories from images.
"""

import os
import torch
import numpy as np
import cv2
import json
import pickle
import warnings
from tqdm import tqdm
from torch import nn

# Suppress common warnings from PyTorch/Detectron2
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

# Import configuration
from config import VCR_IMAGES_DIR

# Set up logger
setup_logger()

class EnhancedFeatureExtractor:
    """
    Enhanced feature extraction class that combines object detection, scene parsing,
    and attribute detection for rich visual representation.
    """
    
    def __init__(self, output_dir, use_cuda=True):
        """
        Initialize the feature extractor with multiple models.
        
        Args:
            output_dir: Directory to save extracted features
            use_cuda: Whether to use CUDA for acceleration
        """
        self.output_dir = output_dir
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initializing Enhanced Feature Extractor (device: {self.device})...")
        
        # Initialize Detectron2 model for object detection and segmentation
        self.setup_detectron2()
        
        print("All models initialized successfully.")
    
    def setup_detectron2(self):
        """Set up Detectron2 with Mask R-CNN for object detection and segmentation"""
        print("Setting up Detectron2 with Mask R-CNN...")
        
        cfg = get_cfg()
        # Using a more powerful X152-FPN backbone for better feature extraction
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detection
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.DEVICE = 'cuda' if self.device.type == 'cuda' else 'cpu'
        
        # Create predictor
        self.detectron_predictor = DefaultPredictor(cfg)
        
        # Get COCO category mapping
        from detectron2.data import MetadataCatalog
        self.coco_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.coco_id_to_name = {k: v for k, v in enumerate(self.coco_metadata.thing_classes)}
        
        print("Detectron2 setup complete.")
        
    def _extract_simple_features(self, image, boxes):
        """
        Simple fallback feature extraction using basic image processing
        """
        features = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Extract crop
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                # Resize to standard size
                crop_resized = cv2.resize(crop, (224, 224))
                # Simple feature: flatten and reduce dimensionality
                feature = crop_resized.flatten()[:2048].astype(np.float32)
                # Pad if necessary
                if len(feature) < 2048:
                    feature = np.pad(feature, (0, 2048 - len(feature)))
                features.append(feature)
            else:
                # Empty crop, use zero features
                features.append(np.zeros(2048, dtype=np.float32))
                
        return np.array(features)
        
    def extract_detectron2_features(self, image, boxes=None, class_ids=None, segms=None):
        """
        Extract features using Detectron2 model
        
        Args:
            image: Input image (numpy array in BGR format)
            boxes: Optional predefined bounding boxes
            class_ids: Optional predefined class IDs
            segms: Optional predefined segmentation masks
            
        Returns:
            Dictionary containing extracted features
        """
        with torch.no_grad():
            # Get raw predictions if no pre-defined boxes
            if boxes is None:
                # Ensure image is in correct format for Detectron2
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                    
                raw_outputs = self.detectron_predictor(image)
                instances = raw_outputs["instances"].to("cpu")
                
                # Use detected boxes
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                
                # Add whole image as a box
                h, w = image.shape[:2]
                boxes = np.vstack([[0, 0, w, h], boxes])
                
                # Get class labels
                class_ids = instances.pred_classes.cpu().numpy()
                class_ids = np.insert(class_ids, 0, -1)  # -1 for whole image
                class_names = [self.coco_id_to_name.get(idx, "unknown") if idx >= 0 else "whole_image" 
                              for idx in class_ids]
                
                # Get detection scores
                scores = instances.scores.cpu().numpy()
                scores = np.insert(scores, 0, 1.0)  # Score 1.0 for whole image
            else:
                # Use provided boxes but still get class predictions
                raw_height, raw_width = image.shape[:2]
                
                # Check if boxes already contain the whole image box at index 0
                has_whole_image = False
                if boxes.shape[0] > 0:
                    # Check if first box approximately covers the whole image
                    first_box = boxes[0]
                    if np.isclose(first_box[0], 0) and np.isclose(first_box[1], 0) and \
                       np.isclose(first_box[2], raw_width, rtol=0.1) and np.isclose(first_box[3], raw_height, rtol=0.1):
                        has_whole_image = True
                
                if not has_whole_image:
                    # Add whole image as first box
                    boxes = np.vstack([[0, 0, raw_width, raw_height], boxes])
                
                # Initialize class information
                class_ids = np.zeros(boxes.shape[0], dtype=np.int64)
                class_ids[0] = -1  # -1 for whole image
                class_names = ["whole_image"] + ["unknown"] * (boxes.shape[0] - 1)
                scores = np.ones(boxes.shape[0])
                
                # If class_ids are provided, use them directly
                if class_ids is not None:
                    # Make sure class_ids is numpy array
                    class_ids = np.array(class_ids)
                    
                    # If first entry isn't whole_image (-1), insert it
                    if len(class_ids) == 0 or class_ids[0] != -1:
                        class_ids = np.insert(class_ids, 0, -1)  # -1 for whole image
                    
                    # Generate class names from IDs
                    class_names = [self.coco_id_to_name.get(idx, "unknown") if idx >= 0 else "whole_image" 
                                  for idx in class_ids]
                else:
                    # Try to predict classes for the provided boxes
                    try:
                        for i in range(1, len(boxes)):
                            box = boxes[i]
                            # Crop the image to the box region
                            x1, y1, x2, y2 = map(int, box)
                            crop = image[y1:y2, x1:x2]
                            
                            if crop.size == 0:  # Skip empty crops
                                continue
                            
                            # Ensure crop is in uint8 format
                            if crop.dtype != np.uint8:
                                crop = crop.astype(np.uint8)
                            
                            # Get predictions for this crop
                            crop_outputs = self.detectron_predictor(crop)
                            crop_instances = crop_outputs["instances"].to("cpu")
                            
                            if len(crop_instances) > 0:
                                # Take the highest scoring detection
                                best_idx = crop_instances.scores.argmax().item()
                                class_ids[i] = crop_instances.pred_classes[best_idx].item()
                                class_names[i] = self.coco_id_to_name.get(class_ids[i], "unknown")
                                scores[i] = crop_instances.scores[best_idx].item()
                    except Exception as e:
                        print(f"Error predicting classes for boxes: {e}")
            
            # Extract features using a simpler approach
            # Ensure image is in uint8 format for Detectron2
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            try:
                # Get predictions first (this should work)
                outputs = self.detectron_predictor(image)
                
                # Extract features from the predictor's last forward pass
                # This is more reliable than manually calling backbone
                if hasattr(self.detectron_predictor.model, 'backbone'):
                    # Use the image preprocessing from detectron2
                    height, width = image.shape[:2]
                    image_preprocessed = self.detectron_predictor.aug.get_transform(image).apply_image(image)
                    # Make a copy to ensure it's writable
                    image_preprocessed = image_preprocessed.copy()
                    image_preprocessed = torch.as_tensor(image_preprocessed.transpose(2, 0, 1))
                    
                    inputs = [{"image": image_preprocessed, "height": height, "width": width}]
                    images = self.detectron_predictor.model.preprocess_image(inputs)
                    
                    # Get backbone features
                    features = self.detectron_predictor.model.backbone(images.tensor)
                    
                    # Convert boxes to the right format and scale
                    box_tensors = torch.as_tensor(boxes, dtype=torch.float32).to(self.device)
                    # Scale boxes to match the feature map
                    if images.tensor.shape[-2:] != (height, width):
                        scale_y = images.tensor.shape[-2] / height
                        scale_x = images.tensor.shape[-1] / width
                        box_tensors = box_tensors * torch.tensor([scale_x, scale_y, scale_x, scale_y]).to(self.device)
                else:
                    # Fallback if backbone access fails
                    raise Exception("Cannot access backbone")
                    
            except Exception as e:
                print(f"Error in backbone feature extraction: {e}, using fallback approach")
                # Create simple CNN-based features as fallback
                box_features = self._extract_simple_features(image, boxes)
                
                # Skip the complex ROI pooling and return simple features
                masks = None
                attributes = {i: [] for i in range(len(boxes))}
                
                return {
                    'image_features': box_features[0] if len(box_features) > 0 else np.zeros(2048),
                    'object_features': box_features[1:] if len(box_features) > 1 else np.array([]),
                    'boxes': boxes,
                    'class_ids': class_ids,
                    'class_names': class_names,
                    'scores': scores,
                    'masks': masks,
                    'attributes': attributes,
                }
            
            try:
                # Get ROI features
                box_features = self.detectron_predictor.model.roi_heads.box_pooler(
                    [features[f] for f in self.detectron_predictor.model.roi_heads.box_in_features],
                    [Boxes(box_tensors)]
                )
                
                # Pass through box head
                box_features = self.detectron_predictor.model.roi_heads.box_head(box_features)
                
                # Convert to numpy
                box_features = box_features.cpu().numpy()
                
            except Exception as e:
                print(f"Error in ROI feature extraction: {e}, using simple features")
                # Create fallback features using simple CNN approach
                box_features = self._extract_simple_features(image, boxes)
            
            # Get segmentation masks if available
            masks = None
            
            # Use provided segmentation masks if available
            if segms is not None:
                # Convert polygon segmentations to binary masks if needed
                h, w = image.shape[:2]
                if isinstance(segms[0], list):  # Polygons
                    binary_masks = []
                    for segm in segms:
                        if isinstance(segm[0], list):  # Multiple polygons for one object
                            mask = np.zeros((h, w), dtype=np.uint8)
                            for polygon in segm:
                                # Convert polygon to numpy array for cv2.fillPoly
                                polygon_np = np.array(polygon, dtype=np.int32).reshape(1, -1, 2)
                                cv2.fillPoly(mask, polygon_np, 1)
                            binary_masks.append(mask.astype(bool))
                        else:
                            # Skip if not a proper polygon
                            continue
                    
                    # Only use masks if we successfully converted some
                    if binary_masks:
                        masks = np.stack(binary_masks)
                
                # Add a whole image mask if we have segmentations
                if masks is not None and len(masks) > 0:
                    whole_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
                    masks = np.vstack([whole_mask[None], masks])
            
            # Fall back to Detectron2 masks if no provided masks and we have instances
            elif hasattr(instances, "pred_masks"):
                masks = instances.pred_masks.cpu().numpy()
                
                # Add a whole image mask
                whole_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
                if len(masks) > 0:
                    masks = np.vstack([whole_mask[None], masks])
                else:
                    masks = whole_mask[None]
            
            # Create attributes dictionary (placeholder for now)
            attributes = {i: [] for i in range(len(boxes))}
            
            # Return all extracted information
            return {
                'image_features': box_features[0],  # Features for the whole image
                'object_features': box_features[1:],  # Features for detected objects
                'boxes': boxes,  # Bounding boxes [x1, y1, x2, y2]
                'class_ids': class_ids,  # Class IDs
                'class_names': class_names,  # Class names
                'scores': scores,  # Detection confidence scores
                'masks': masks,  # Segmentation masks (if available)
                'attributes': attributes,  # Visual attributes (placeholder)
            }

    def process_image(self, image_path, metadata_path=None, output_path=None):
        """
        Process a single image and extract features
        
        Args:
            image_path: Path to the input image
            metadata_path: Optional path to metadata JSON with predefined boxes
            output_path: Path to save extracted features
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                return False
            
            # Ensure image is in correct format (uint8)
            if image.dtype != np.uint8:
                # Convert to uint8 if it's in a different format
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # If image values are in [0, 1] range, scale to [0, 255]
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Load metadata if available
            boxes = None
            class_ids = None
            segms = None
            
            if metadata_path and os.path.exists(metadata_path):
                try:
                    metadata = json.load(open(metadata_path))
                    
                    # Get boxes if available
                    if 'boxes' in metadata:
                        boxes = np.array(metadata['boxes'])[:,:4]
                    
                    # Get class names/IDs if available
                    if 'names' in metadata:
                        # Map VCR class names to COCO class IDs where possible
                        vcr_names = metadata.get('names', [])
                        class_ids = []
                        
                        # Create a mapping from COCO class names to IDs
                        coco_name_to_id = {v: k for k, v in self.coco_id_to_name.items()}
                        
                        for name in vcr_names:
                            # Try direct mapping first
                            if name in coco_name_to_id:
                                class_ids.append(coco_name_to_id[name])
                            else:
                                # Handle common cases
                                if name == 'person':
                                    class_ids.append(0)  # person in COCO
                                else:
                                    # Add as unknown class
                                    class_ids.append(-100)  # Use -100 for unknown classes
                    
                    # Get segmentation masks if available
                    if 'segms' in metadata:
                        segms = metadata['segms']
                
                except Exception as e:
                    print(f"Error loading metadata: {e}")
            
            # Extract features
            features = self.extract_detectron2_features(image, boxes, class_ids, segms)
            
            # Save features
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}.pkl")
                
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save features
            with open(output_path, 'wb') as f:
                pickle.dump(features, f)
                
            return True
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False

    def process_directory(self, input_dir, recursive=True):
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory containing images
            recursive: Whether to process subdirectories recursively
        """
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_path = os.path.join(root, file)
                        image_files.append(image_path)
        else:
            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(input_dir, file)
                    image_files.append(image_path)
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        # Process each image
        successful = 0
        for image_path in tqdm(image_files, desc="Processing images"):
            # Check for metadata file
            base_path = os.path.splitext(image_path)[0]
            metadata_path = base_path + '.json'
            if not os.path.exists(metadata_path):
                metadata_path = None
                
            # Construct output path
            rel_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(self.output_dir, os.path.splitext(rel_path)[0] + '.pkl')
            
            # Process image
            if self.process_image(image_path, metadata_path, output_path):
                successful += 1
                
        print(f"Successfully processed {successful}/{len(image_files)} images")

def main():
    """Main function to run feature extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Feature Extraction for Visual Backstory Generation")
    parser.add_argument("--input_dir", default=VCR_IMAGES_DIR, 
                        help="Input directory containing images")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory to save features")
    parser.add_argument("--image_path", 
                        help="Path to a single image (optional)")
    parser.add_argument("--metadata_path",
                        help="Path to metadata JSON (optional, for single image)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Process subdirectories recursively")
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = EnhancedFeatureExtractor(args.output_dir, use_cuda=not args.no_cuda)
    
    if args.image_path:
        # Process a single image
        print(f"Processing single image: {args.image_path}")
        extractor.process_image(args.image_path, args.metadata_path)
    else:
        # Process all images in the input directory
        print(f"Processing images from directory: {args.input_dir}")
        extractor.process_directory(args.input_dir, recursive=args.recursive)
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()
