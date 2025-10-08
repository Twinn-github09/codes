"""
Updated inference script for the unified Visual Backstory Generation model.
Compatible with your trained unified model system.
"""

import os
import sys
import json
import argparse
import logging
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Add parent directory to path so we can import from models and dataloaders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import unified model system
from models.unified_model import UnifiedVisualBackstoryModel, TransformerType, create_unified_config
from scripts.image_to_backstory import ImageToBackstoryPipeline

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def generate_backstory_from_checkpoint(checkpoint_path, image_path, place=None, event=None, 
                                     device="cuda", num_samples=1, max_length=50, temperature=0.9):
    """
    Generate backstory using your trained unified model
    
    Args:
        checkpoint_path: Path to your trained model checkpoint
        image_path: Path to the input image
        place: Optional place description (auto-detected if None)
        event: Optional event description (auto-detected if None)
        device: Device to run inference on
        num_samples: Number of backstories to generate
        max_length: Maximum length of generated text
        temperature: Sampling temperature
    
    Returns:
        Dictionary with results
    """
    # Initialize pipeline with your trained model
    pipeline = ImageToBackstoryPipeline(checkpoint_path, device=device)
    
    if place is None or event is None:
        # Use complete pipeline for scene understanding
        result = pipeline.process_image(image_path, max_length=max_length)
        return {
            'image_path': image_path,
            'detected_place': result['place'],
            'detected_event': result['event'],
            'backstory': result['backstory'],
            'method': 'auto_detected'
        }
    else:
        # Use manual place/event descriptions
        # Extract features for manual generation
        image_features = pipeline.extract_image_features(image_path)
        visual_features_dict = pipeline.prepare_visual_features_for_backstory_model(image_features)
        
        # Generate backstory with specified place/event
        backstory = pipeline.generate_backstory(visual_features_dict, place, event, max_length)
        
        return {
            'image_path': image_path,
            'specified_place': place,
            'specified_event': event,
            'backstory': backstory,
            'method': 'manual'
        }

def batch_inference(checkpoint_path, image_dir, output_file, device="cuda", num_samples=1):
    """
    Run inference on a directory of images using your trained model
    
    Args:
        checkpoint_path: Path to your trained model checkpoint
        image_dir: Directory containing input images
        output_file: Path to save generated backstories
        device: Device to run inference on
        num_samples: Number of backstories per image (currently supports 1)
    """
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                          if f.lower().endswith(ext)])
    
    logger.info(f"Found {len(image_files)} images in {image_dir}")
    
    # Generate backstories
    results = {}
    
    for image_path in tqdm(image_files, desc="Generating backstories"):
        image_name = os.path.basename(image_path)
        
        try:
            # Generate backstory using complete pipeline
            result = generate_backstory_from_checkpoint(
                checkpoint_path, image_path, device=device, 
                num_samples=num_samples
            )
            
            results[image_name] = result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results[image_name] = {
                'error': str(e),
                'image_path': image_path
            }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def interactive_demo(checkpoint_path, device="cuda"):
    """
    Run an interactive demo with your trained model
    
    Args:
        checkpoint_path: Path to your trained model checkpoint
        device: Device to run inference on
    """
    # Initialize pipeline
    logger.info("Loading trained model...")
    pipeline = ImageToBackstoryPipeline(checkpoint_path, device=device)
    
    logger.info("🎬 Interactive Backstory Generation Demo (Your Trained Model)")
    logger.info("Enter 'q' to quit")
    
    while True:
        print("\n" + "="*50)
        # Get image path from user
        image_path = input("📸 Enter image path: ").strip()
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            logger.error(f"❌ Image not found: {image_path}")
            continue
        
        # Ask for mode
        print("\nChoose mode:")
        print("1. Auto-detect place and event (recommended)")
        print("2. Manually specify place and event")
        mode = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
        
        try:
            if mode == "1":
                # Auto-detection mode
                result = pipeline.process_image(image_path)
                
                print(f"\n🎯 RESULTS:")
                print(f"📍 Detected Place: {result['place']}")
                print(f"🎭 Detected Event: {result['event']}")
                print(f"📖 Generated Backstory: {result['backstory']}")
                
            else:
                # Manual mode
                place = input("📍 Enter place description: ").strip()
                event = input("🎭 Enter event description: ").strip()
                
                if not place or not event:
                    logger.error("❌ Both place and event are required for manual mode")
                    continue
                
                result = generate_backstory_from_checkpoint(
                    checkpoint_path, image_path, place=place, event=event, device=device
                )
                
                print(f"\n🎯 RESULTS:")
                print(f"📍 Place: {result['specified_place']}")
                print(f"🎭 Event: {result['specified_event']}")
                print(f"📖 Generated Backstory: {result['backstory']}")
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")

def main():
    """Parse arguments and run the script"""
    parser = argparse.ArgumentParser(description="Inference for Your Trained Visual Backstory Model")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to your trained model checkpoint (.pt file)")
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], default="interactive",
                        help="Inference mode")
    parser.add_argument("--image_path",
                        help="Path to input image (for single mode)")
    parser.add_argument("--image_dir",
                        help="Directory containing input images (for batch mode)")
    parser.add_argument("--output_file",
                        help="Path to save generated backstories (for batch mode)")
    parser.add_argument("--place",
                        help="Place description (for single mode)")
    parser.add_argument("--event",
                        help="Event description (for single mode)")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated backstory")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    logger.info(f"Using device: {device}")
    
    if not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if args.mode == "single":
        if args.image_path is None:
            parser.error("--image_path is required for single mode")
            
        result = generate_backstory_from_checkpoint(
            args.checkpoint, args.image_path,
            place=args.place, event=args.event, device=device,
            max_length=args.max_length
        )
        
        print(f"\n🎯 RESULTS:")
        if 'detected_place' in result:
            print(f"📍 Detected Place: {result['detected_place']}")
            print(f"🎭 Detected Event: {result['detected_event']}")
        else:
            print(f"📍 Place: {result['specified_place']}")
            print(f"🎭 Event: {result['specified_event']}")
        print(f"📖 Generated Backstory: {result['backstory']}")
            
    elif args.mode == "batch":
        if args.image_dir is None or args.output_file is None:
            parser.error("--image_dir and --output_file are required for batch mode")
            
        batch_inference(
            args.checkpoint, args.image_dir, args.output_file, device=device
        )
        
    else:  # interactive
        interactive_demo(args.checkpoint, device=device)

if __name__ == "__main__":
    main()