"""
Inference script for Visual Backstory Generation model.
This script loads a trained model and generates backstories for input images.
"""

import os
import json
import pickle
import argparse
import logging
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Import custom modules
from dataloaders.enhanced_feature_extraction import EnhancedFeatureExtractor
from models.backstory_model import VisualBackstoryGenerationModel
from dataloaders.tokenizers import VisualCometTokenizer

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def generate_backstory(model, tokenizer, feature_extractor, image_path, place=None, event=None, 
                      device="cuda", num_samples=1, max_length=50, temperature=0.9, top_p=0.9):
    """
    Generate backstory for a single image
    
    Args:
        model: Trained backstory generation model
        tokenizer: Tokenizer for text processing
        feature_extractor: Feature extraction model
        image_path: Path to the input image
        place: Optional place description
        event: Optional event description
        device: Device to run inference on
        num_samples: Number of backstories to generate
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        List of generated backstories
    """
    # Extract features
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    features = feature_extractor.extract_detectron2_features(image)
    
    # Convert features to tensors
    visual_inputs = {
        'image_features': torch.tensor(features['image_features'], dtype=torch.float).unsqueeze(0).to(device),
        'object_features': torch.tensor(features['object_features'], dtype=torch.float).unsqueeze(0).to(device),
        'boxes': torch.tensor(features['boxes'], dtype=torch.float).unsqueeze(0).to(device),
        'box_mask': torch.ones(1, features['object_features'].shape[0] + 1, dtype=torch.long).to(device),
        'class_ids': torch.tensor(features['class_ids'], dtype=torch.long).unsqueeze(0).to(device)
    }
    
    # Create text prompt
    if place is None:
        place = "unknown place"
    if event is None:
        event = "unknown event"
        
    text_input = f"Place: {place} Event: {event} Before: "
    input_ids = tokenizer.encode(text_input, return_tensors="pt").to(device)
    
    # Generate backstories
    with torch.no_grad():
        # Encode visual features
        mean, logvar = model.encode(visual_inputs)
        
        # Generate multiple samples if requested
        backstories = []
        for _ in range(num_samples):
            # Sample from latent space
            if num_samples > 1:
                latent_z = model.reparameterize(mean, logvar)
            else:
                # Use mean for single deterministic generation
                latent_z = mean
            
            # Generate text
            generated_ids = model.generate(
                visual_features=visual_inputs,
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=(num_samples > 1),
                num_return_sequences=1
            )
            
            # Decode generated text
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            backstory = generated_text[len(text_input):]
            backstories.append(backstory)
    
    return backstories


def batch_inference(model_path, image_dir, output_file, device="cuda", num_samples=3):
    """
    Run inference on a directory of images
    
    Args:
        model_path: Path to the trained model
        image_dir: Directory containing input images
        output_file: Path to save generated backstories
        device: Device to run inference on
        num_samples: Number of backstories to generate per image
    """
    # Load model and tokenizer
    tokenizer = VisualCometTokenizer.from_pretrained(model_path)
    model = VisualBackstoryGenerationModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor(output_dir="temp_features", use_cuda=(device=="cuda"))
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    logger.info(f"Found {len(image_files)} images in {image_dir}")
    
    # Generate backstories
    results = {}
    
    for image_path in tqdm(image_files, desc="Generating backstories"):
        image_name = os.path.basename(image_path)
        
        try:
            # Generate backstories
            backstories = generate_backstory(
                model, tokenizer, feature_extractor, image_path,
                device=device, num_samples=num_samples
            )
            
            results[image_name] = {
                'image_path': image_path,
                'backstories': backstories
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def interactive_demo(model_path, device="cuda"):
    """
    Run an interactive demo where the user can input images and get generated backstories
    
    Args:
        model_path: Path to the trained model
        device: Device to run inference on
    """
    # Load model and tokenizer
    tokenizer = VisualCometTokenizer.from_pretrained(model_path)
    model = VisualBackstoryGenerationModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor(output_dir="temp_features", use_cuda=(device=="cuda"))
    
    logger.info("Interactive Backstory Generation Demo")
    logger.info("Enter 'q' to quit")
    
    while True:
        # Get image path from user
        image_path = input("\nEnter image path: ")
        if image_path.lower() == 'q':
            break
            
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            continue
        
        # Get optional place and event descriptions
        place = input("Enter place description (or leave empty): ")
        event = input("Enter event description (or leave empty): ")
        
        # Set defaults if empty
        place = place if place else None
        event = event if event else None
        
        # Get generation parameters
        try:
            num_samples = int(input("Number of backstories to generate (default: 3): ") or 3)
            temperature = float(input("Temperature (default: 0.9): ") or 0.9)
        except ValueError:
            logger.error("Invalid input, using defaults")
            num_samples = 3
            temperature = 0.9
        
        try:
            # Generate backstories
            backstories = generate_backstory(
                model, tokenizer, feature_extractor, image_path,
                place=place, event=event, device=device,
                num_samples=num_samples, temperature=temperature
            )
            
            # Print generated backstories
            logger.info("\nGenerated Backstories:")
            for i, backstory in enumerate(backstories):
                logger.info(f"[{i+1}] {backstory}")
                
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Parse arguments and run the script"""
    parser = argparse.ArgumentParser(description="Inference for Visual Backstory Generation")
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained model")
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
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of backstories to generate per image")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    if args.mode == "single":
        # Run inference on a single image
        if args.image_path is None:
            parser.error("--image_path is required for single mode")
            
        # Load model and tokenizer
        tokenizer = VisualCometTokenizer.from_pretrained(args.model_path)
        model = VisualBackstoryGenerationModel.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
        
        # Initialize feature extractor
        feature_extractor = EnhancedFeatureExtractor(output_dir="temp_features", use_cuda=(device=="cuda"))
        
        # Generate backstories
        backstories = generate_backstory(
            model, tokenizer, feature_extractor, args.image_path,
            place=args.place, event=args.event, device=device,
            num_samples=args.num_samples, temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Print generated backstories
        logger.info("\nGenerated Backstories:")
        for i, backstory in enumerate(backstories):
            logger.info(f"[{i+1}] {backstory}")
            
    elif args.mode == "batch":
        # Run inference on a directory of images
        if args.image_dir is None or args.output_file is None:
            parser.error("--image_dir and --output_file are required for batch mode")
            
        batch_inference(
            args.model_path, args.image_dir, args.output_file,
            device=device, num_samples=args.num_samples
        )
        
    else:  # interactive
        # Run interactive demo
        interactive_demo(args.model_path, device=device)


if __name__ == "__main__":
    main()
