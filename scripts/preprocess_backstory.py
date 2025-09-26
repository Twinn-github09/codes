"""
Preprocessing Pipeline for Visual Backstory Generation.
This script preprocesses images and annotations for backstory generation training.
"""

import sys
import os

# Add parent directory to path so we can import from dataloaders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import time
from datetime import datetime

# Import our enhanced feature extraction
from dataloaders.enhanced_feature_extraction import EnhancedFeatureExtractor

# Import configuration
from config import VCR_IMAGES_DIR

class BackstoryPreprocessor:
    """
    Preprocessor class for preparing data for visual backstory generation.
    This class handles:
    1. Extracting visual features from images
    2. Processing annotations to focus on 'before' events
    3. Creating training, validation, and test splits
    """
    
    def __init__(self, input_dir, output_dir, vcr_images_dir=VCR_IMAGES_DIR, use_cuda=True, checkpoint_interval=100):
        """
        Initialize the preprocessor
        
        Args:
            input_dir: Directory containing Visual COMET annotations
            output_dir: Directory to save preprocessed data
            vcr_images_dir: Directory containing VCR images
            use_cuda: Whether to use CUDA for feature extraction
            checkpoint_interval: Save checkpoint every N processed images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.vcr_images_dir = vcr_images_dir
        self.use_cuda = use_cuda
        self.checkpoint_interval = checkpoint_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Features output directory
        self.features_dir = os.path.join(output_dir, 'features')
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Processed annotations directory
        self.processed_dir = os.path.join(output_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Checkpoint file paths
        self.progress_checkpoint = os.path.join(self.checkpoint_dir, 'feature_extraction_progress.json')
        self.annotations_checkpoint = os.path.join(self.checkpoint_dir, 'processed_annotations.json')
        
        # Initialize feature extractor
        self.feature_extractor = EnhancedFeatureExtractor(self.features_dir, use_cuda=use_cuda)
        
        print(f"Initialized BackstoryPreprocessor:")
        print(f"  - Input dir: {input_dir}")
        print(f"  - Output dir: {output_dir}")
        print(f"  - VCR images dir: {vcr_images_dir}")
        print(f"  - Checkpoint interval: {checkpoint_interval} images")
        print(f"  - Checkpoint dir: {self.checkpoint_dir}")
    
    def save_progress_checkpoint(self, processed_images, skipped_existing, skipped_missing, total_images):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'skipped_existing': skipped_existing,
                'skipped_missing': skipped_missing,
                'total_images': total_images,
                'processed_image_list': list(processed_images)  # Convert set to list for JSON serialization
            }
            
            with open(self.progress_checkpoint, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")
            # Try to save a minimal checkpoint
            try:
                minimal_checkpoint = {
                    'timestamp': datetime.now().isoformat(),
                    'skipped_existing': int(skipped_existing),
                    'skipped_missing': int(skipped_missing),
                    'total_images': int(total_images),
                    'processed_image_list': []  # Empty list as fallback
                }
                with open(self.progress_checkpoint, 'w') as f:
                    json.dump(minimal_checkpoint, f, indent=2)
                print("Saved minimal checkpoint without processed image list")
            except Exception as e2:
                print(f"Error: Could not save even minimal checkpoint: {e2}")
    
    def load_progress_checkpoint(self):
        """Load progress from checkpoint file"""
        if os.path.exists(self.progress_checkpoint):
            try:
                with open(self.progress_checkpoint, 'r') as f:
                    checkpoint_data = json.load(f)
                
                processed_images = set(checkpoint_data.get('processed_image_list', []))
                skipped_existing = checkpoint_data.get('skipped_existing', 0)
                skipped_missing = checkpoint_data.get('skipped_missing', 0)
                timestamp = checkpoint_data.get('timestamp', 'Unknown')
                
                print(f"📁 Found checkpoint from {timestamp}")
                print(f"   - Previously processed: {len(processed_images)} images")
                print(f"   - Previously skipped (existing): {skipped_existing}")
                print(f"   - Previously skipped (missing): {skipped_missing}")
                
                return processed_images, skipped_existing, skipped_missing
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                return set(), 0, 0
        else:
            print("📁 No previous checkpoint found, starting from beginning")
            return set(), 0, 0
    
    def save_annotations_checkpoint(self, annotations):
        """Save processed annotations to checkpoint"""
        with open(self.annotations_checkpoint, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"💾 Saved annotations checkpoint with {sum(len(split_data) for split_data in annotations.values())} total entries")
    
    def load_annotations_checkpoint(self):
        """Load processed annotations from checkpoint"""
        if os.path.exists(self.annotations_checkpoint):
            try:
                with open(self.annotations_checkpoint, 'r') as f:
                    annotations = json.load(f)
                print(f"📁 Loaded annotations checkpoint with {sum(len(split_data) for split_data in annotations.values())} total entries")
                return annotations
            except Exception as e:
                print(f"Warning: Could not load annotations checkpoint: {e}")
                return None
        return None
    
    def process_annotations(self):
        """
        Process Visual COMET annotations to focus on backstory generation.
        Only keep the 'before' events and discard 'intent' and 'after' events.
        """
        # Load annotations for each split
        splits = ['train', 'val', 'test']
        all_annotations = {}
        
        for split in splits:
            annotation_file = os.path.join(self.input_dir, f'{split}_annots.json')
            if not os.path.exists(annotation_file):
                print(f"Warning: Annotation file not found: {annotation_file}")
                continue
                
            print(f"Processing {split} annotations...")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Filter and process annotations
            backstory_annotations = []
            
            for i, ann in enumerate(tqdm(annotations, desc=f"Processing {split} annotations")):
                # Skip entries without 'before' events
                if 'before' not in ann or not ann['before']:
                    continue
                    
                # Create new entry focused on backstory
                backstory_entry = {
                    'id': f"{split}_{i}",
                    'img_fn': ann['img_fn'],
                    'movie': ann.get('movie', ''),
                    'metadata_fn': ann.get('metadata_fn', ''),
                    'place': ann.get('place', ''),
                    'event': ann.get('event', ''),
                    'before': ann['before'],  # List of backstory events
                }
                
                backstory_annotations.append(backstory_entry)
            
            print(f"Kept {len(backstory_annotations)} out of {len(annotations)} entries with valid backstory")
            
            # Save processed annotations
            output_file = os.path.join(self.processed_dir, f'{split}_backstory.json')
            with open(output_file, 'w') as f:
                json.dump(backstory_annotations, f, indent=2)
                
            all_annotations[split] = backstory_annotations
            
        return all_annotations
    
    def extract_all_features(self, annotations):
        """
        Extract visual features for all images in the annotations with checkpoint support
        
        Args:
            annotations: Dictionary of annotations by split
        """
        # Collect all unique image paths
        image_paths = set()
        metadata_paths = {}
        
        for split, split_annotations in annotations.items():
            for ann in split_annotations:
                img_fn = ann['img_fn']
                metadata_fn = ann.get('metadata_fn', '')
                
                # Extract movie and image ID
                parts = img_fn.split('/')
                if len(parts) >= 2:
                    movie, img_id = parts[0], parts[1]
                    
                    # Construct full image path
                    img_path = os.path.join(self.vcr_images_dir, movie, img_id)
                    image_paths.add(img_path)
                    
                    # Add metadata path if available
                    if metadata_fn:
                        metadata_path = os.path.join(self.vcr_images_dir, metadata_fn)
                        metadata_paths[img_path] = metadata_path
        
        print(f"Found {len(image_paths)} unique images to process")
        
        # Load checkpoint if exists
        previously_processed, skipped_existing, skipped_missing = self.load_progress_checkpoint()
        
        # Track statistics
        processed_count = len(previously_processed)
        checkpoint_counter = 0
        
        # Convert to list for easier iteration with progress tracking
        image_paths_list = list(image_paths)
        
        # Process each image
        for i, img_path in enumerate(tqdm(image_paths_list, desc="Extracting features")):
            metadata_path = metadata_paths.get(img_path, None)
            
            # Construct output path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            movie_name = os.path.basename(os.path.dirname(img_path))
            output_path = os.path.join(self.features_dir, f"{movie_name}_{base_name}.pkl")
            
            # Skip if already processed in previous run
            if img_path in previously_processed:
                continue
            
            # Skip if already processed in current run
            if os.path.exists(output_path):
                skipped_existing += 1
                continue
            
            # Check if image file exists before processing
            if not os.path.exists(img_path):
                skipped_missing += 1
                tqdm.write(f"Skipping missing image: {img_path}")
                continue
                
            # Process image - the process_image method will return False if it fails
            try:
                success = self.feature_extractor.process_image(img_path, metadata_path, output_path)
                if success:
                    processed_count += 1
                    previously_processed.add(img_path)
                    checkpoint_counter += 1
                    
                    # Save checkpoint periodically
                    if checkpoint_counter >= self.checkpoint_interval:
                        self.save_progress_checkpoint(
                            previously_processed, skipped_existing, skipped_missing, len(image_paths_list)
                        )
                        checkpoint_counter = 0
                        tqdm.write(f"💾 Checkpoint saved at {processed_count} processed images")
                else:
                    skipped_missing += 1
                    tqdm.write(f"Failed to process image: {img_path}")
            except Exception as e:
                skipped_missing += 1
                tqdm.write(f"Error processing {img_path}: {str(e)}")
        
        # Save final checkpoint
        self.save_progress_checkpoint(
            previously_processed, skipped_existing, skipped_missing, len(image_paths_list)
        )
        
        print(f"\nFeature extraction complete:")
        print(f"  - Total processed: {processed_count}")
        print(f"  - Skipped (already existed): {skipped_existing}")
        print(f"  - Skipped (missing/failed): {skipped_missing}")
        print(f"  - Total images: {len(image_paths_list)}")
        print(f"💾 Final checkpoint saved")
    
    def create_dataset_files(self, annotations):
        """
        Create final dataset files for training
        
        Args:
            annotations: Dictionary of annotations by split
        """
        for split, split_annotations in annotations.items():
            dataset = []
            skipped_count = 0
            
            for ann in tqdm(split_annotations, desc=f"Creating {split} dataset"):
                # Extract movie and image ID from img_fn
                img_fn = ann['img_fn']
                parts = img_fn.split('/')
                
                if len(parts) >= 2:
                    movie, img_id = parts[0], parts[1]
                    base_name = os.path.splitext(img_id)[0]
                    
                    # Construct feature path
                    feature_path = os.path.join(self.features_dir, f"{movie}_{base_name}.pkl")
                    
                    # Check if features exist
                    if not os.path.exists(feature_path):
                        skipped_count += 1
                        continue
                    
                    # Choose one backstory sentence randomly (or keep all, depending on your approach)
                    # For this implementation, we keep all backstory sentences as separate examples
                    for i, backstory in enumerate(ann['before']):
                        example = {
                            'id': f"{ann['id']}_{i}",
                            'img_fn': img_fn,
                            'feature_path': os.path.relpath(feature_path, self.output_dir),
                            'place': ann['place'],
                            'event': ann['event'],
                            'backstory': backstory,
                        }
                        
                        dataset.append(example)
            
            # Save dataset
            output_path = os.path.join(self.processed_dir, f'{split}_dataset.json')
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
                
            print(f"Created {split} dataset with {len(dataset)} examples (skipped {skipped_count} entries due to missing features)")
    
    def run_preprocessing(self, resume_from_checkpoint=True):
        """
        Run the entire preprocessing pipeline with checkpoint support
        
        Args:
            resume_from_checkpoint: Whether to resume from existing checkpoints
        """
        print("🚀 Starting preprocessing pipeline...")
        
        # Check for annotation checkpoint first
        if resume_from_checkpoint:
            annotations = self.load_annotations_checkpoint()
            
            if annotations is not None:
                print("📁 Resuming from annotation checkpoint")
            else:
                print("📋 Processing annotations from scratch...")
                annotations = self.process_annotations()
                self.save_annotations_checkpoint(annotations)
        else:
            print("📋 Processing annotations from scratch (ignoring checkpoints)...")
            annotations = self.process_annotations()
            self.save_annotations_checkpoint(annotations)
        
        # Extract features for all images (with automatic checkpoint support)
        print("🎯 Starting feature extraction...")
        self.extract_all_features(annotations)
        
        # Create dataset files
        print("📊 Creating final dataset files...")
        self.create_dataset_files(annotations)
        
        print("✅ Preprocessing complete!")
        print(f"📁 Checkpoints saved in: {self.checkpoint_dir}")
        
    def clean_checkpoints(self):
        """Clean all checkpoint files"""
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print("🧹 All checkpoints cleaned")
        
    def get_progress_status(self):
        """Get current progress status"""
        if os.path.exists(self.progress_checkpoint):
            with open(self.progress_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            processed = len(checkpoint_data.get('processed_image_list', []))
            skipped_existing = checkpoint_data.get('skipped_existing', 0)
            skipped_missing = checkpoint_data.get('skipped_missing', 0)
            total = checkpoint_data.get('total_images', 0)
            timestamp = checkpoint_data.get('timestamp', 'Unknown')
            
            print(f"📊 Current Progress Status:")
            print(f"   Last updated: {timestamp}")
            print(f"   Processed: {processed}/{total} images")
            print(f"   Skipped (existing): {skipped_existing}")
            print(f"   Skipped (missing): {skipped_missing}")
            
            if total > 0:
                progress_percent = (processed + skipped_existing + skipped_missing) / total * 100
                print(f"   Overall progress: {progress_percent:.1f}%")
            
            return {
                'processed': processed,
                'skipped_existing': skipped_existing,
                'skipped_missing': skipped_missing,
                'total': total,
                'timestamp': timestamp
            }
        else:
            print("📊 No progress checkpoint found")
            return None

def main():
    """Main function to run preprocessing"""
    parser = argparse.ArgumentParser(description="Preprocess data for Visual Backstory Generation with Checkpoint Support")
    parser.add_argument("--input_dir", required=True,
                        help="Input directory containing Visual COMET annotations")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--vcr_images_dir", default=VCR_IMAGES_DIR,
                        help="Directory containing VCR images")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save checkpoint every N processed images (default: 100)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from checkpoint, start fresh")
    parser.add_argument("--status", action="store_true",
                        help="Show current progress status and exit")
    parser.add_argument("--clean_checkpoints", action="store_true",
                        help="Clean all checkpoints and start fresh")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = BackstoryPreprocessor(
        args.input_dir,
        args.output_dir,
        args.vcr_images_dir,
        use_cuda=not args.no_cuda,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Handle different modes
    if args.status:
        # Show status and exit
        preprocessor.get_progress_status()
        return
    
    if args.clean_checkpoints:
        # Clean checkpoints and exit
        preprocessor.clean_checkpoints()
        return
    
    # Run preprocessing
    resume_from_checkpoint = not args.no_resume
    preprocessor.run_preprocessing(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    main()
