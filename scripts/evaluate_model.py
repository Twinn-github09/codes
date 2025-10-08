"""
Evaluate or use trained Visual Backstory Generation model for inference
"""

import os
import sys
import argparse
import torch
import pickle
from pathlib import Path

# Add the visual-comet directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_model import UnifiedVisualBackstoryModel, TransformerType
from models.config import create_unified_config
from transformers import GPT2Tokenizer
import numpy as np

def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on
        
    Returns:
        model, config, tokenizer
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Recreate config
    config_dict = checkpoint['config']
    transformer_type = TransformerType(checkpoint['transformer_type'])
    
    # Create model
    config = create_unified_config(
        vocab_size=config_dict['vocab_size'],
        transformer_type=transformer_type,
        **{k: v for k, v in config_dict.items() if k != 'vocab_size'}
    )
    
    model = UnifiedVisualBackstoryModel(config, transformer_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Transformer type: {transformer_type.value}")
    print(f"Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, config, tokenizer, checkpoint

def load_visual_features(feature_path):
    """Load visual features from pickle file"""
    try:
        with open(feature_path, 'rb') as f:
            features_dict = pickle.load(f)
        
        # Extract features based on file structure
        if 'features' in features_dict:
            visual_features = features_dict['features']
        elif 'image_features' in features_dict:
            visual_features = features_dict['image_features']
        else:
            visual_features = features_dict
        
        # Ensure numpy array
        if isinstance(visual_features, torch.Tensor):
            visual_features = visual_features.numpy()
        elif not isinstance(visual_features, np.ndarray):
            visual_features = np.array(visual_features)
        
        return visual_features
    except Exception as e:
        print(f"Error loading features: {e}")
        return None

def prepare_visual_features_for_model(visual_features, device, target_dim=2048, num_boxes=36):
    """
    Prepare visual features in the format expected by the model
    
    Args:
        visual_features: Raw visual features
        device: Target device
        target_dim: Target feature dimension (2048 for detector)
        num_boxes: Number of object boxes
        
    Returns:
        Dictionary with visual features ready for model
    """
    if visual_features is None:
        # Create dummy features
        features = np.zeros((num_boxes, target_dim), dtype=np.float32)
        print("Using dummy visual features")
    else:
        # Reshape and project features
        if len(visual_features.shape) == 1:
            # Single feature vector
            original_dim = visual_features.shape[0]
            features = np.zeros((num_boxes, target_dim), dtype=np.float32)
            
            if original_dim <= target_dim:
                features[0, :original_dim] = visual_features
            else:
                features[0] = visual_features[:target_dim]
        else:
            # Multiple features
            original_dim = visual_features.shape[1]
            num_features = min(visual_features.shape[0], num_boxes)
            features = np.zeros((num_boxes, target_dim), dtype=np.float32)
            
            for i in range(num_features):
                if original_dim <= target_dim:
                    features[i, :original_dim] = visual_features[i]
                else:
                    features[i] = visual_features[i, :target_dim]
    
    # Create visual features dictionary
    visual_features_dict = {
        'image_features': torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device),  # Add batch dim
        'boxes': torch.zeros(1, num_boxes, 4, dtype=torch.float32).to(device),  # Dummy boxes
        'box_mask': torch.zeros(1, num_boxes, dtype=torch.long).to(device),     # Dummy mask
        'class_ids': torch.zeros(1, num_boxes, dtype=torch.long).to(device)    # Dummy class IDs
    }
    
    # Mark first box as valid
    visual_features_dict['box_mask'][0, 0] = 1
    visual_features_dict['boxes'][0, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0])  # Full image box
    
    return visual_features_dict

def generate_backstory(model, tokenizer, visual_features_dict, place_text, event_text, device, max_length=100):
    """
    Generate backstory given visual features and text context
    
    Args:
        model: Trained model
        tokenizer: GPT2 tokenizer
        visual_features_dict: Prepared visual features
        place_text: Place description
        event_text: Event description
        device: Device
        max_length: Maximum generation length
        
    Returns:
        Generated backstory text
    """
    # Create input text
    input_text = f"Place: {place_text} Event: {event_text}"
    print(f"Input: {input_text}")
    
    # Tokenize input
    input_tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # Generate
        if hasattr(model, 'generate'):
            # Use model's generate method if available
            output_tokens = model.generate(
                visual_features=visual_features_dict,
                input_ids=input_tokens,
                max_length=input_tokens.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            # Simple generation using model forward pass
            outputs = model(
                visual_features=visual_features_dict,
                input_ids=input_tokens,
                attention_mask=None,
                labels=None
            )
            
            # Get next token probabilities and sample
            logits = outputs['logits'][0, -1, :]  # Last token logits
            probs = torch.softmax(logits / 0.8, dim=-1)  # Temperature sampling
            next_token = torch.multinomial(probs, 1)
            
            # Simple continuation (just add one token for demo)
            output_tokens = torch.cat([input_tokens, next_token.unsqueeze(0)], dim=1)
    
    # Decode output
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    # Extract the generated part (after input)
    if input_text in generated_text:
        backstory = generated_text[len(input_text):].strip()
    else:
        backstory = generated_text.strip()
    
    return backstory

def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual Backstory Generation Model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--visual_features", help="Path to visual features pickle file")
    parser.add_argument("--place", default="in a park", help="Place description")
    parser.add_argument("--event", default="a person is walking", help="Event description")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config, tokenizer, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load visual features if provided
    visual_features = None
    if args.visual_features and os.path.exists(args.visual_features):
        visual_features = load_visual_features(args.visual_features)
        print(f"Loaded visual features from: {args.visual_features}")
        if visual_features is not None:
            print(f"Visual features shape: {visual_features.shape}")
    else:
        print("No visual features provided, using dummy features")
    
    # Prepare visual features for model
    visual_features_dict = prepare_visual_features_for_model(visual_features, device)
    
    # Generate backstory
    print(f"\\n=== GENERATING BACKSTORY ===")
    backstory = generate_backstory(
        model, tokenizer, visual_features_dict, 
        args.place, args.event, device, args.max_length
    )
    
    print(f"\\nGenerated backstory: {backstory}")
    print("===========================")

if __name__ == "__main__":
    main()