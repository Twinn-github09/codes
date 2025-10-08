"""
Enhanced inference pipeline using original feature extraction methods
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_model import UnifiedVisualBackstoryModel, TransformerType, create_unified_config
from transformers import GPT2Tokenizer

class OriginalFeatureBackstoryPipeline:
    """
    Backstory generation using original feature extraction methods
    (same as training data)
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.backstory_model, self.tokenizer = self._load_backstory_model(checkpoint_path)
        
        # Common place/event combinations from training data
        self.common_scenarios = [
            ("in a kitchen", "someone is cooking"),
            ("in a kitchen", "someone is eating"), 
            ("in a kitchen", "someone is cleaning"),
            ("outdoors", "a person is walking"),
            ("outdoors", "people are talking"),
            ("in a living room", "someone is watching TV"),
            ("in a living room", "people are sitting"),
            ("in an office", "someone is working"),
            ("in a bedroom", "a person is sleeping"),
            ("in a car", "a person is driving"),
            ("at a restaurant", "someone is eating"),
            ("in a store", "people are shopping")
        ]
    
    def _load_backstory_model(self, checkpoint_path):
        """Load the trained backstory generation model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Recreate model
        config_dict = checkpoint['config']
        transformer_type = TransformerType(checkpoint['transformer_type'])
        
        filtered_config = {k: v for k, v in config_dict.items() 
                          if k not in ['vocab_size', 'transformer_type']}
        
        config = create_unified_config(
            vocab_size=config_dict['vocab_size'],
            transformer_type=transformer_type,
            **filtered_config
        )
        
        model = UnifiedVisualBackstoryModel(config, transformer_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Backstory model loaded successfully!")
        print(f"Epoch: {checkpoint['epoch']}, Type: {transformer_type.value}")
        
        return model, tokenizer
    
    def load_original_features(self, feature_path):
        """
        Load original preprocessed features (same as training)
        
        Args:
            feature_path: Path to .pkl feature file
            
        Returns:
            Visual features dictionary
        """
        try:
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)
            
            # Extract the visual features (same format as training)
            if isinstance(features, dict):
                # Handle different feature file formats
                if 'features' in features:
                    img_features = features['features']
                elif 'image_features' in features:
                    img_features = features['image_features']
                else:
                    # Assume the dict values contain the features
                    img_features = list(features.values())[0]
            else:
                img_features = features
            
            # Ensure proper shape (should be 2048-dim)
            if img_features.shape[-1] != 2048:
                print(f"Warning: Feature dimension is {img_features.shape[-1]}, expected 2048")
            
            # Create visual features dictionary for model
            visual_features_dict = {
                'image_features': torch.tensor(img_features, dtype=torch.float32).unsqueeze(0).to(self.device),
                'boxes': torch.zeros(1, img_features.shape[0], 4, dtype=torch.float32).to(self.device),
                'box_mask': torch.ones(1, img_features.shape[0], dtype=torch.long).to(self.device),
                'class_ids': torch.zeros(1, img_features.shape[0], dtype=torch.long).to(self.device)
            }
            
            return visual_features_dict
            
        except Exception as e:
            print(f"Error loading features from {feature_path}: {e}")
            return None
    
    def generate_with_original_features(self, feature_path, place=None, event=None, max_length=50):
        """
        Generate backstory using original feature file
        
        Args:
            feature_path: Path to .pkl feature file (same format as training)
            place: Optional place description
            event: Optional event description
            max_length: Maximum backstory length
            
        Returns:
            Generated backstory
        """
        # Load original features
        visual_features_dict = self.load_original_features(feature_path)
        if visual_features_dict is None:
            return "Error loading visual features"
        
        # Use provided place/event or default to common scenario
        if place is None or event is None:
            place, event = self.common_scenarios[0]  # Default to kitchen/cooking
            
        return self.generate_backstory(visual_features_dict, place, event, max_length)
    
    def generate_backstory(self, visual_features_dict, place, event, max_length=50):
        """
        Generate backstory using the trained CVAE model
        """
        # Create input prompt
        input_text = f"Place: {place} Event: {event}"
        print(f"\\nGenerating backstory for: {input_text}")
        
        # Tokenize input
        input_tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            try:
                # Generate using model
                generated_tokens = input_tokens.clone()
                
                for _ in range(max_length):
                    outputs = self.backstory_model(
                        visual_features=visual_features_dict,
                        input_ids=generated_tokens,
                        attention_mask=None,
                        labels=None
                    )
                    
                    # Get next token
                    logits = outputs['logits'][0, -1, :]
                    probs = torch.softmax(logits / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # Stop if EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Decode
                generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Extract backstory
                if input_text in generated_text:
                    backstory = generated_text[len(input_text):].strip()
                else:
                    backstory = generated_text.strip()
                
                return backstory if backstory else "The story begins..."
                
            except Exception as e:
                print(f"Error generating backstory: {e}")
                return f"Before this moment, they prepared carefully..."
    
    def batch_generate_with_scenarios(self, feature_path, num_scenarios=3):
        """
        Generate multiple backstories with different place/event combinations
        """
        visual_features_dict = self.load_original_features(feature_path)
        if visual_features_dict is None:
            return []
        
        results = []
        for i, (place, event) in enumerate(self.common_scenarios[:num_scenarios]):
            print(f"\\n--- Scenario {i+1}: {place} + {event} ---")
            backstory = self.generate_backstory(visual_features_dict, place, event)
            results.append({
                'place': place,
                'event': event,
                'backstory': backstory
            })
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = OriginalFeatureBackstoryPipeline('models/filtered/normal_backstory_model_latest.pt')
    
    # Test with original feature file
    feature_file = "path/to/your/feature.pkl"  # Your .pkl files
    
    # Generate with specific scenario
    result = pipeline.generate_with_original_features(
        feature_file, 
        place="in a kitchen", 
        event="someone is cooking"
    )
    print(f"Generated backstory: {result}")
    
    # Generate multiple scenarios
    scenarios = pipeline.batch_generate_with_scenarios(feature_file, num_scenarios=3)
    for scenario in scenarios:
        print(f"{scenario['place']} + {scenario['event']}: {scenario['backstory']}")