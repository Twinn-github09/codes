"""
Enhanced backstory generation pipeline using:
1. Your trained feature extractor for visual features
2. External API models for scene understanding (place/event detection)
3. Your CVAE model for backstory generation
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import requests
import json
import base64
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_model import UnifiedVisualBackstoryModel, TransformerType, create_unified_config
from transformers import GPT2Tokenizer
from dataloaders.enhanced_feature_extraction import EnhancedFeatureExtractor

class APIEnhancedBackstoryPipeline:
    """
    Enhanced pipeline using external APIs for better scene understanding
    """
    
    def __init__(self, checkpoint_path, device='cuda', use_api='openai'):
        """
        Initialize the enhanced pipeline
        
        Args:
            checkpoint_path: Path to trained backstory model
            device: Device to run on
            use_api: Which API to use ('openai', 'google', 'azure', 'huggingface')
        """
        self.device = device
        self.use_api = use_api
        
        print(f"Using device: {self.device}")
        print(f"Using API: {self.use_api}")
        
        # Load backstory generation model
        print("Loading backstory model...")
        self.backstory_model, self.tokenizer = self._load_backstory_model(checkpoint_path)
        
        # Initialize your trained feature extractor
        print("Initializing feature extractor...")
        self.feature_extractor = EnhancedFeatureExtractor(
            output_dir="temp_features", 
            use_cuda=(device=="cuda")
        )
        
        # API configurations
        self.api_configs = {
            'openai': {
                'url': 'https://api.openai.com/v1/chat/completions',
                'headers': {
                    'Authorization': 'Bearer YOUR_OPENAI_API_KEY',
                    'Content-Type': 'application/json'
                }
            },
            'google': {
                'url': 'https://vision.googleapis.com/v1/images:annotate',
                'key': 'YOUR_GOOGLE_API_KEY'
            },
            'azure': {
                'url': 'https://YOUR_RESOURCE.cognitiveservices.azure.com/vision/v3.2/analyze',
                'key': 'YOUR_AZURE_KEY'
            },
            'huggingface': {
                'url': 'https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large',
                'headers': {
                    'Authorization': 'Bearer YOUR_HF_TOKEN'
                }
            }
        }
    
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
    
    def extract_visual_features(self, image_path):
        """
        Extract visual features using your trained feature extractor
        (Same quality as training data)
        """
        try:
            print(f"\\nExtracting visual features from: {image_path}")
            
            # Use your enhanced feature extractor
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract features using your trained pipeline
            features = self.feature_extractor.extract_detectron2_features(image)
            
            # Convert to model format
            visual_features_dict = {
                'image_features': torch.tensor(features['image_features'], dtype=torch.float).unsqueeze(0).to(self.device),
                'boxes': torch.tensor(features['boxes'], dtype=torch.float).unsqueeze(0).to(self.device),
                'box_mask': torch.ones(1, features['boxes'].shape[0], dtype=torch.long).to(self.device),
                'class_ids': torch.tensor(features['class_ids'], dtype=torch.long).unsqueeze(0).to(self.device)
            }
            
            print(f"✅ Visual features extracted: {features['image_features'].shape}")
            return visual_features_dict
            
        except Exception as e:
            print(f"❌ Error extracting visual features: {e}")
            return None
    
    def analyze_scene_with_api(self, image_path):
        """
        Use external API for better scene understanding
        """
        try:
            print(f"\\nAnalyzing scene with {self.use_api.upper()} API...")
            
            if self.use_api == 'openai':
                return self._analyze_with_openai(image_path)
            elif self.use_api == 'google':
                return self._analyze_with_google(image_path)
            elif self.use_api == 'azure':
                return self._analyze_with_azure(image_path)
            elif self.use_api == 'huggingface':
                return self._analyze_with_huggingface(image_path)
            else:
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"❌ API analysis failed: {e}")
            return self._fallback_analysis()
    
    def _analyze_with_openai(self, image_path):
        """Analyze scene using OpenAI GPT-4 Vision"""
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this image and provide:
1. PLACE: Where is this scene taking place? (e.g., 'in a kitchen', 'outdoors', 'in an office')
2. EVENT: What is currently happening? (e.g., 'someone is cooking', 'a person is walking')

Format your response as:
PLACE: [location]
EVENT: [current action]"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }
        
        response = requests.post(
            self.api_configs['openai']['url'],
            headers=self.api_configs['openai']['headers'],
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            return self._parse_api_response(content)
        else:
            raise Exception(f"OpenAI API error: {response.status_code}")
    
    def _analyze_with_huggingface(self, image_path):
        """Analyze scene using Hugging Face models"""
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Get image caption first
        response = requests.post(
            self.api_configs['huggingface']['url'],
            headers=self.api_configs['huggingface']['headers'],
            data=image_data
        )
        
        if response.status_code == 200:
            result = response.json()
            caption = result[0]['generated_text'] if result else ""
            
            # Extract place and event from caption
            place, event = self._extract_place_event_from_caption(caption)
            return {"place": place, "event": event, "caption": caption}
        else:
            raise Exception(f"Hugging Face API error: {response.status_code}")
    
    def _parse_api_response(self, content):
        """Parse API response to extract place and event"""
        lines = content.strip().split('\\n')
        place = "somewhere"
        event = "something is happening"
        
        for line in lines:
            if line.startswith('PLACE:'):
                place = line.replace('PLACE:', '').strip()
            elif line.startswith('EVENT:'):
                event = line.replace('EVENT:', '').strip()
        
        return {"place": place, "event": event}
    
    def _extract_place_event_from_caption(self, caption):
        """Extract place and event from image caption"""
        # Simple heuristic extraction (can be improved)
        caption_lower = caption.lower()
        
        # Common places
        places = {
            'kitchen': 'in a kitchen',
            'bedroom': 'in a bedroom',
            'living room': 'in a living room',
            'office': 'in an office',
            'outdoor': 'outdoors',
            'street': 'on a street',
            'park': 'in a park',
            'restaurant': 'at a restaurant',
            'car': 'in a car'
        }
        
        # Common events
        events = {
            'cooking': 'someone is cooking',
            'eating': 'someone is eating',
            'walking': 'a person is walking',
            'sitting': 'a person is sitting',
            'working': 'someone is working',
            'reading': 'someone is reading',
            'driving': 'a person is driving'
        }
        
        place = "somewhere"
        event = "something is happening"
        
        # Find place
        for key, value in places.items():
            if key in caption_lower:
                place = value
                break
        
        # Find event
        for key, value in events.items():
            if key in caption_lower:
                event = value
                break
        
        return place, event
    
    def _fallback_analysis(self):
        """Fallback analysis when API fails"""
        return {
            "place": "somewhere interesting",
            "event": "something meaningful is happening"
        }
    
    def generate_backstory(self, visual_features_dict, place, event, max_length=50):
        """
        Generate backstory using your trained CVAE model
        """
        input_text = f"Place: {place} Event: {event}"
        print(f"\\nGenerating backstory for: {input_text}")
        
        input_tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            try:
                generated_tokens = input_tokens.clone()
                
                for _ in range(max_length):
                    outputs = self.backstory_model(
                        visual_features=visual_features_dict,
                        input_ids=generated_tokens,
                        attention_mask=None,
                        labels=None
                    )
                    
                    logits = outputs['logits'][0, -1, :]
                    probs = torch.softmax(logits / 0.8, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                
                generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                if input_text in generated_text:
                    backstory = generated_text[len(input_text):].strip()
                else:
                    backstory = generated_text.strip()
                
                return backstory if backstory else "The story begins here..."
                
            except Exception as e:
                print(f"Error generating backstory: {e}")
                return f"Before {event}, they prepared for this moment..."
    
    def process_image_with_api(self, image_path, max_length=50):
        """
        Complete pipeline: Image -> Features + API Scene Analysis -> Backstory
        """
        print(f"\\n{'='*60}")
        print(f"PROCESSING IMAGE: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Step 1: Extract visual features using your trained extractor
        visual_features_dict = self.extract_visual_features(image_path)
        if visual_features_dict is None:
            return {"error": "Failed to extract visual features"}
        
        # Step 2: Analyze scene using external API
        scene_analysis = self.analyze_scene_with_api(image_path)
        
        # Step 3: Generate backstory using your CVAE model
        backstory = self.generate_backstory(
            visual_features_dict, 
            scene_analysis['place'], 
            scene_analysis['event'], 
            max_length
        )
        
        result = {
            'image_path': image_path,
            'place': scene_analysis['place'],
            'event': scene_analysis['event'],
            'backstory': backstory,
            'api_used': self.use_api
        }
        
        print(f"\\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"{'='*60}")
        print(f"📍 Place: {result['place']}")
        print(f"🎭 Event: {result['event']}")
        print(f"📖 Backstory: {result['backstory']}")
        print(f"🤖 API Used: {result['api_used']}")
        print(f"{'='*60}")
        
        return result

# Interactive demo
def interactive_demo():
    """Interactive demo with API-enhanced scene understanding"""
    print("🎬 API-Enhanced Backstory Generation")
    print("=" * 60)
    
    # Choose API
    print("\\nAvailable APIs:")
    print("1. OpenAI GPT-4 Vision (requires API key)")
    print("2. Hugging Face BLIP (free)")
    print("3. Local fallback (no API)")
    
    api_choice = input("\\nChoose API (1-3, default: 2): ").strip() or "2"
    api_map = {"1": "openai", "2": "huggingface", "3": "local"}
    use_api = api_map.get(api_choice, "huggingface")
    
    # Initialize pipeline
    checkpoint_path = "models/filtered/normal_backstory_model_latest.pt"
    pipeline = APIEnhancedBackstoryPipeline(checkpoint_path, use_api=use_api)
    
    while True:
        print("\\n" + "-" * 60)
        image_path = input("📸 Enter image path (or 'quit'): ").strip()
        
        if image_path.lower() in ['quit', 'q', 'exit']:
            break
            
        if not os.path.exists(image_path):
            print("❌ Image not found!")
            continue
        
        try:
            result = pipeline.process_image_with_api(image_path)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_demo()