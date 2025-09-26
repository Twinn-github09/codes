#!/usr/bin/env python
"""
Interactive test script for Visual Backstory Generation.
This script provides an interactive interface to test your trained model.
"""

import os
import sys
import argparse
import torch
import logging
import glob
from pathlib import Path

# Import necessary modules
from scripts.generate_backstory import interactive_demo

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def find_latest_model(checkpoint_dir):
    """
    Find the most recent model checkpoint in the specified directory
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        
    Returns:
        Path to the most recent model checkpoint directory
    """
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None
    
    # Sort by modification time
    latest_dir = max(checkpoint_dirs, key=os.path.getmtime)
    return latest_dir

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Interactive Test for Visual Backstory Generation")
    parser.add_argument("--model_path", 
                        help="Path to the trained model directory. If not provided, will look for the latest checkpoint.")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory containing model checkpoints (used if model_path not provided)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Find model path if not provided
    model_path = args.model_path
    if model_path is None:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        model_path = find_latest_model(checkpoint_dir)
        if model_path is None:
            logger.error(f"No model checkpoints found in {checkpoint_dir}")
            logger.info("Please train the model first or provide a valid --model_path")
            sys.exit(1)
        logger.info(f"Using latest model checkpoint: {model_path}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    # Start interactive demo
    logger.info("Starting interactive backstory generation demo")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Using device: {device}")
    logger.info("-------------------------------------")
    interactive_demo(model_path, device=device)

if __name__ == "__main__":
    main()
