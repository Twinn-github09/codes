"""
End-to-end Image-to-Backstory Generation Pipeline
Takes raw image as input and generates backstory automatically
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer
from transformers import CLIPProcessor, CLIPModel

# Add the visual-comet directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_model import UnifiedVisualBackstoryModel, TransformerType
from models.unified_model import create_unified_config

class ImageToBackstoryPipeline:
    """Complete pipeline from image to backstory"""
    
    def __init__(self, model_checkpoint_path, device='cuda'):
        """
        Initialize the pipeline
        
        Args:
            model_checkpoint_path: Path to trained backstory model
            device: Device to run on
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load CLIP for image understanding
        print("Loading CLIP model for image understanding...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load backstory generation model
        print(f"Loading backstory model from: {model_checkpoint_path}")
        self.backstory_model, self.tokenizer = self._load_backstory_model(model_checkpoint_path)
        
        # Define place and event categories for scene understanding
        self.place_templates = [
            "in a park", "in a restaurant", "at home", "in an office", "on a street",
            "in a kitchen", "in a bedroom", "in a living room", "in a bathroom",
            "in a garden", "at a beach", "in a car", "on a bus", "in a train",
            "in a store", "in a mall", "in a school", "in a hospital",
            "in a library", "in a gym", "outdoors", "indoors"
        ]
        
        self.event_templates = [
            "people are talking", "someone is eating", "a person is walking",
            "people are working", "someone is reading", "a person is sleeping",
            "people are playing", "someone is cooking", "a person is driving",
            "people are shopping", "someone is exercising", "a person is studying",
            "people are dancing", "someone is cleaning", "a person is sitting",
            "people are running", "someone is watching TV", "a person is using phone",
            "people are laughing", "someone is drinking", "a person is thinking"
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
        
        # Filter out keys that shouldn't be passed to create_unified_config
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
    
    def extract_image_features(self, image_path):
        """
        Extract features from image using CLIP
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image features as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features.cpu().numpy().flatten()
            
            print(f"Extracted image features: {image_features.shape}")
            return image_features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def understand_scene(self, image_path):
        """
        Understand the scene (place and event) from the image using CLIP
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_place, predicted_event)
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                # Get image features
                image_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compare with place templates
                place_inputs = self.clip_processor(text=self.place_templates, return_tensors="pt", padding=True).to(self.device)
                place_features = self.clip_model.get_text_features(**place_inputs)
                place_features = place_features / place_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities for places
                place_similarities = (image_features @ place_features.T).squeeze(0)
                best_place_idx = torch.argmax(place_similarities).item()
                best_place = self.place_templates[best_place_idx]
                
                # Compare with event templates
                event_inputs = self.clip_processor(text=self.event_templates, return_tensors="pt", padding=True).to(self.device)
                event_features = self.clip_model.get_text_features(**event_inputs)
                event_features = event_features / event_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities for events
                event_similarities = (image_features @ event_features.T).squeeze(0)
                best_event_idx = torch.argmax(event_similarities).item()
                best_event = self.event_templates[best_event_idx]
                
                print(f"Scene Understanding:")
                print(f"  Place: {best_place} (confidence: {place_similarities[best_place_idx]:.3f})")
                print(f"  Event: {best_event} (confidence: {event_similarities[best_event_idx]:.3f})")
                
                return best_place, best_event
                
        except Exception as e:
            print(f"Error understanding scene: {e}")
            return "somewhere", "something is happening"
    
    def prepare_visual_features_for_backstory_model(self, image_features):
        """
        Convert CLIP features to format expected by backstory model
        
        Args:
            image_features: CLIP image features
            
        Returns:
            Visual features dictionary for backstory model
        """
        if image_features is None:
            # Create dummy features
            features = np.zeros((36, 2048), dtype=np.float32)
        else:
            # CLIP features are 512-dim, need to project to 2048-dim
            clip_dim = image_features.shape[0]  # Should be 512
            target_dim = 2048
            num_boxes = 36
            
            # Create features matrix
            features = np.zeros((num_boxes, target_dim), dtype=np.float32)
            
            # Project CLIP features to first part of feature vector
            if clip_dim <= target_dim:
                features[0, :clip_dim] = image_features
            else:
                features[0] = image_features[:target_dim]
        
        # Create visual features dictionary
        visual_features_dict = {
            'image_features': torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device),
            'boxes': torch.zeros(1, 36, 4, dtype=torch.float32).to(self.device),
            'box_mask': torch.zeros(1, 36, dtype=torch.long).to(self.device),
            'class_ids': torch.zeros(1, 36, dtype=torch.long).to(self.device)
        }
        
        # Mark first box as valid (whole image)
        visual_features_dict['box_mask'][0, 0] = 1
        visual_features_dict['boxes'][0, 0] = torch.tensor([0.0, 0.0, 1.0, 1.0])
        
        return visual_features_dict
    
    def generate_backstory(self, visual_features_dict, place, event, max_length=50):
        """
        Generate backstory using the trained model
        
        Args:
            visual_features_dict: Prepared visual features
            place: Detected place
            event: Detected event
            max_length: Maximum generation length
            
        Returns:
            Generated backstory text
        """
        # Create input prompt
        input_text = f"Place: {place} Event: {event}"
        print(f"\\nGenerating backstory for: {input_text}")
        
        # Tokenize input
        input_tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            try:
                # Simple generation using model forward pass
                # Generate multiple tokens iteratively
                generated_tokens = input_tokens.clone()
                
                for _ in range(max_length):
                    outputs = self.backstory_model(
                        visual_features=visual_features_dict,
                        input_ids=generated_tokens,
                        attention_mask=None,
                        labels=None
                    )
                    
                    # Get next token probabilities
                    logits = outputs['logits'][0, -1, :]
                    probs = torch.softmax(logits / 0.8, dim=-1)  # Temperature sampling
                    next_token = torch.multinomial(probs, 1)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    # Append next token
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                # Decode generated text
                generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Extract backstory (remove input part)
                if input_text in generated_text:
                    backstory = generated_text[len(input_text):].strip()
                else:
                    backstory = generated_text.strip()
                
                return backstory if backstory else "The story begins here..."
                
            except Exception as e:
                print(f"Error generating backstory: {e}")
                return f"Before {event.replace('someone', 'they').replace('a person', 'they')}, they prepared for this moment."
    
    def process_image(self, image_path, max_length=50):
        """
        Complete pipeline: image -> features -> scene understanding -> backstory
        
        Args:
            image_path: Path to input image
            max_length: Maximum backstory length
            
        Returns:
            Dictionary with all results
        """
        print(f"\\n=== PROCESSING IMAGE: {os.path.basename(image_path)} ===")
        
        # Step 1: Extract image features
        print("\\n1. Extracting image features...")
        image_features = self.extract_image_features(image_path)
        
        # Step 2: Understand scene
        print("\\n2. Understanding scene...")
        place, event = self.understand_scene(image_path)
        
        # Step 3: Prepare features for backstory model
        print("\\n3. Preparing features for backstory model...")
        visual_features_dict = self.prepare_visual_features_for_backstory_model(image_features)
        
        # Step 4: Generate backstory
        print("\\n4. Generating backstory...")
        backstory = self.generate_backstory(visual_features_dict, place, event, max_length)
        
        results = {
            'image_path': image_path,
            'place': place,
            'event': event,
            'backstory': backstory,
            'input_prompt': f"Place: {place} Event: {event}"
        }
        
        print(f"\\n=== RESULTS ===")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Detected Place: {place}")
        print(f"Detected Event: {event}")
        print(f"Generated Backstory: {backstory}")
        print("================")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate backstory from image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum backstory length")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found!")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found!")
        return
    
    # Initialize pipeline
    try:
        pipeline = ImageToBackstoryPipeline(args.checkpoint, args.device)
        
        # Process image
        results = pipeline.process_image(args.image, args.max_length)
        
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        print("Make sure you have CLIP installed: pip install clip-by-openai")

if __name__ == "__main__":
    main()