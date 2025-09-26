#!/usr/bin/env python
"""
Simple test script for the Visual Backstory Generation model.
This script provides an easy way to test a trained model with any image.
"""

import os
import argparse
import logging
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Import necessary modules
from scripts.generate_backstory import generate_backstory
from dataloaders.enhanced_feature_extraction import EnhancedFeatureExtractor
from models.backstory_model import VisualBackstoryGenerationModel
from dataloaders.tokenizers import VisualCometTokenizer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def show_image_with_backstory(image_path, backstories):
    """
    Display the image with generated backstories
    
    Args:
        image_path: Path to the image
        backstories: List of generated backstories
    """
    # Load and display image
    img = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    
    # Add backstories as title
    backstory_text = "\n\n".join([f"Backstory {i+1}: {story}" for i, story in enumerate(backstories)])
    plt.title(backstory_text, fontsize=12, wrap=True)
    
    plt.tight_layout()
    plt.show()

def test_model(model_path, image_path, place=None, event=None, num_samples=3, 
               temperature=0.9, top_p=0.9, visualize=True, save_path=None):
    """
    Test a trained backstory generation model on a single image
    
    Args:
        model_path: Path to the trained model directory
        image_path: Path to the test image
        place: Optional place description
        event: Optional event description
        num_samples: Number of backstories to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        visualize: Whether to visualize the results
        save_path: Path to save the results
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    try:
        tokenizer = VisualCometTokenizer.from_pretrained(model_path)
        model = VisualBackstoryGenerationModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Make sure you've trained the model first using train_backstory.py")
        return
    
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor(output_dir="temp_features", use_cuda=(device=="cuda"))
    
    # Generate backstories
    logger.info(f"Generating backstories for {image_path}")
    backstories = generate_backstory(
        model, tokenizer, feature_extractor, image_path,
        place=place, event=event, device=device,
        num_samples=num_samples, temperature=temperature,
        top_p=top_p
    )
    
    # Print generated backstories
    logger.info("\nGenerated Backstories:")
    for i, backstory in enumerate(backstories):
        logger.info(f"[{i+1}] {backstory}")
    
    # Visualize if requested
    if visualize:
        show_image_with_backstory(image_path, backstories)
    
    # Save results if requested
    if save_path:
        with open(save_path, 'w') as f:
            for i, backstory in enumerate(backstories):
                f.write(f"Backstory {i+1}: {backstory}\n")
        logger.info(f"Results saved to {save_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test Visual Backstory Generation Model")
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--image_path", required=True,
                        help="Path to the test image")
    parser.add_argument("--place",
                        help="Optional place description")
    parser.add_argument("--event",
                        help="Optional event description")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of backstories to generate")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Don't visualize the results")
    parser.add_argument("--save_path",
                        help="Path to save the results")
    
    args = parser.parse_args()
    
    # Test the model
    test_model(
        model_path=args.model_path,
        image_path=args.image_path,
        place=args.place,
        event=args.event,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        visualize=not args.no_visualize,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()
