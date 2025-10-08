"""
Simple enhanced backstory generation using your feature extractor + local models
No API keys required - uses local Hugging Face models
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unified_model import UnifiedVisualBackstoryModel, TransformerType, create_unified_config
from transformers import GPT2Tokenizer, BlipProcessor, BlipForConditionalGeneration
from dataloaders.enhanced_feature_extraction import EnhancedFeatureExtractor
import google.generativeai as genai
import base64
import json
from scripts.gemini_postprocessor import GeminiBackstoryPostProcessor

class LocalEnhancedBackstoryPipeline:
    """
    Enhanced pipeline using local models for scene understanding
    """
    
    def __init__(self, checkpoint_path, device='cuda', gemini_api_key=None):
        # Auto-detect available device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("⚠️  CUDA not available, using CPU")
        
        self.device = device
        self.gemini_api_key = gemini_api_key
        
        print(f"Using device: {self.device}")
        
        # Load backstory generation model
        print("Loading backstory model...")
        self.backstory_model, self.tokenizer = self._load_backstory_model(checkpoint_path)
        
        # Detect model type from checkpoint path
        if 'bayesian' in checkpoint_path.lower():
            self.model_type = 'bayesian'
        else:
            self.model_type = 'normal'
        print(f"   Model type detected: {self.model_type}")
        
        # Initialize your trained feature extractor  
        print("Initializing your trained feature extractor...")
        self.feature_extractor = EnhancedFeatureExtractor(
            output_dir="temp_features", 
            use_cuda=(device=="cuda")
        )
        
        # Initialize Gemini Post-Processor for presentation enhancement
        self.gemini_postprocessor = None
        if gemini_api_key:
            try:
                self.gemini_postprocessor = GeminiBackstoryPostProcessor(api_key=gemini_api_key)
                print("✅ Gemini Post-Processor enabled (for presentation quality)")
            except Exception as e:
                print(f"⚠️  Could not initialize Gemini Post-Processor: {e}")
        
        # Initialize Gemini API
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                # Try working model names (based on API check)
                model_names = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash', 'gemini-pro-latest']
                self.gemini_model = None
                
                for model_name in model_names:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        print(f"✅ Gemini API initialized successfully with {model_name}")
                        break
                    except Exception as model_e:
                        print(f"⚠️  {model_name} not available: {model_e}")
                        continue
                
                if self.gemini_model is None:
                    print("❌ No Gemini models available, falling back to BLIP")
                    
            except Exception as e:
                print(f"❌ Warning: Could not initialize Gemini API: {e}")
                self.gemini_model = None
        else:
            print("ℹ️  No Gemini API key provided, using fallback BLIP model")
            self.gemini_model = None
        
        # Load BLIP as fallback for scene understanding
        if not self.gemini_model:
            print("Loading BLIP model for scene analysis...")
            try:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
                print("✅ BLIP model loaded successfully")
            except Exception as e:
                print(f"❌ Warning: Could not load BLIP model: {e}")
                self.blip_processor = None
                self.blip_model = None
        else:
            self.blip_processor = None
            self.blip_model = None
    
    def _load_backstory_model(self, checkpoint_path):
        """Load your trained backstory generation model"""
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
        
        print(f"✅ Backstory model loaded successfully!")
        print(f"   Epoch: {checkpoint['epoch']}, Type: {transformer_type.value}")
        
        return model, tokenizer
    
    def extract_trained_features(self, image_path):
        """
        Extract visual features using YOUR trained feature extractor
        (Same quality and format as training data)
        """
        try:
            print(f"\\n🔍 Extracting features with your trained extractor...")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract features using your enhanced feature extractor
            features = self.feature_extractor.extract_detectron2_features(image)
            
            print(f"🔧 Raw feature shapes:")
            for key, value in features.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
            
            # Fix feature formatting to match training expectations
            image_features = features['image_features']
            
            # Handle different feature formats
            if len(image_features.shape) == 1:
                # Single feature vector - expand to multiple regions
                if image_features.shape[0] == 1024:
                    # Split 1024 into 36 regions of ~28 features each, then pad to 2048
                    num_regions = 36
                    features_per_region = image_features.shape[0] // num_regions
                    
                    # Reshape and pad to 2048 dimensions
                    reshaped_features = image_features[:num_regions * features_per_region].reshape(num_regions, features_per_region)
                    padded_features = torch.zeros(num_regions, 2048)
                    padded_features[:, :features_per_region] = torch.tensor(reshaped_features, dtype=torch.float)
                    image_features = padded_features
                else:
                    # Create 36 regions by repeating/splitting the feature
                    image_features = torch.tensor(image_features, dtype=torch.float).repeat(36, 1)
                    if image_features.shape[1] != 2048:
                        # Pad or truncate to 2048
                        if image_features.shape[1] < 2048:
                            padding = torch.zeros(36, 2048 - image_features.shape[1])
                            image_features = torch.cat([image_features, padding], dim=1)
                        else:
                            image_features = image_features[:, :2048]
            
            # Ensure boxes format
            if 'boxes' not in features or features['boxes'] is None:
                # Create default bounding boxes (36 regions in 6x6 grid)
                boxes = []
                for i in range(6):
                    for j in range(6):
                        x1, y1 = j * 0.16, i * 0.16
                        x2, y2 = (j + 1) * 0.16, (i + 1) * 0.16
                        boxes.append([x1, y1, x2, y2])
                boxes = torch.tensor(boxes, dtype=torch.float)
            else:
                boxes = torch.tensor(features['boxes'], dtype=torch.float)
                if boxes.shape[0] != 36:
                    # Pad or truncate to 36 boxes
                    if boxes.shape[0] < 36:
                        # Repeat last box to fill 36
                        last_box = boxes[-1:] if len(boxes) > 0 else torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                        padding_needed = 36 - boxes.shape[0]
                        padding_boxes = last_box.repeat(padding_needed, 1)
                        boxes = torch.cat([boxes, padding_boxes], dim=0)
                    else:
                        boxes = boxes[:36]
            
            # Ensure class_ids format
            if 'class_ids' not in features or features['class_ids'] is None:
                class_ids = torch.zeros(36, dtype=torch.long)
            else:
                class_ids = torch.tensor(features['class_ids'], dtype=torch.long)
                if class_ids.shape[0] != 36:
                    if class_ids.shape[0] < 36:
                        padding_needed = 36 - class_ids.shape[0]
                        padding_ids = torch.zeros(padding_needed, dtype=torch.long)
                        class_ids = torch.cat([class_ids, padding_ids], dim=0)
                    else:
                        class_ids = class_ids[:36]
            
            # Convert to model format (same as training)
            visual_features_dict = {
                'image_features': image_features.unsqueeze(0).to(self.device),  # [1, 36, 2048]
                'boxes': boxes.unsqueeze(0).to(self.device),                    # [1, 36, 4]
                'box_mask': torch.ones(1, 36, dtype=torch.long).to(self.device), # [1, 36]
                'class_ids': class_ids.unsqueeze(0).to(self.device)            # [1, 36]
            }
            
            print(f"✅ Features formatted for model:")
            for key, tensor in visual_features_dict.items():
                print(f"   {key}: {tensor.shape}")
            
            return visual_features_dict
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            print(f"🔄 Using fallback dummy features...")
            # Fallback to dummy features
            return self._create_dummy_features()
    
    def _create_dummy_features(self):
        """Create dummy features as fallback"""
        print("⚠️  Using dummy features as fallback...")
        return {
            'image_features': torch.zeros(1, 36, 2048, dtype=torch.float).to(self.device),
            'boxes': torch.zeros(1, 36, 4, dtype=torch.float).to(self.device),
            'box_mask': torch.zeros(1, 36, dtype=torch.long).to(self.device),
            'class_ids': torch.zeros(1, 36, dtype=torch.long).to(self.device)
        }
    
    def analyze_scene_with_gemini(self, image_path):
        """
        Analyze scene using Gemini API for smart understanding
        """
        try:
            if self.gemini_model is None:
                return self.analyze_scene_with_blip(image_path)
            
            print(f"🚀 Analyzing scene with Gemini...")
            
            # Load and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create prompt for scene analysis
            prompt = """
            Analyze this image and provide a JSON response with exactly this format:
            {
                "place": "where this is happening (e.g., 'in a kitchen', 'on a table', 'at a park')",
                "event": "what is happening (e.g., 'a cat is resting', 'someone is cooking', 'people are walking')",
                "caption": "brief description of the scene"
            }
            
            Be specific about the location and action. Focus on the main subject and setting.
            """
            
            # Upload image and analyze
            image_part = {
                "mime_type": "image/jpeg",
                "data": image_data
            }
            
            response = self.gemini_model.generate_content([prompt, image_part])
            
            # Parse JSON response
            try:
                # Clean the response text
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                scene_data = json.loads(response_text)
                
                place = scene_data.get('place', 'somewhere')
                event = scene_data.get('event', 'something is happening')
                caption = scene_data.get('caption', 'Scene analysis')
                
                print(f"📝 Gemini caption: {caption}")
                print(f"📍 Gemini place: {place}")
                print(f"🎭 Gemini event: {event}")
                
                return {"place": place, "event": event, "caption": caption}
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse Gemini JSON response: {e}")
                print(f"Raw response: {response.text}")
                # Fallback to simple parsing
                return self._parse_gemini_fallback(response.text)
            
        except Exception as e:
            print(f"❌ Gemini analysis failed: {e}")
            return self.analyze_scene_with_blip(image_path)
    
    def _parse_gemini_fallback(self, response_text):
        """Fallback parsing when JSON fails"""
        try:
            lines = response_text.lower().split('\n')
            place = "somewhere"
            event = "something is happening"
            caption = response_text[:100] + "..." if len(response_text) > 100 else response_text
            
            # Simple keyword extraction from response
            for line in lines:
                if 'place' in line or 'location' in line or 'where' in line:
                    if ':' in line:
                        place = line.split(':', 1)[1].strip(' "')
                elif 'event' in line or 'action' in line or 'happening' in line:
                    if ':' in line:
                        event = line.split(':', 1)[1].strip(' "')
                elif 'caption' in line or 'description' in line:
                    if ':' in line:
                        caption = line.split(':', 1)[1].strip(' "')
            
            return {"place": place, "event": event, "caption": caption}
        except:
            return self._fallback_scene_analysis()

    def analyze_scene_with_blip(self, image_path):
        """
        Analyze scene using BLIP model as fallback
        """
        try:
            if self.blip_model is None:
                return self._fallback_scene_analysis()
            
            print(f"🧠 Analyzing scene with BLIP...")
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=120)
                caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Extract place and event from caption
            place, event = self._extract_place_event_from_caption(caption)
            
            print(f"📝 Generated caption: {caption}")
            print(f"📍 Extracted place: {place}")
            print(f"🎭 Extracted event: {event}")
            
            return {"place": place, "event": event, "caption": caption}
            
        except Exception as e:
            print(f"❌ BLIP analysis failed: {e}")
            return self._fallback_scene_analysis()
    
    def _extract_place_event_from_caption(self, caption):
        """Extract place and event from image caption using keyword matching"""
        caption_lower = caption.lower()
        
        # Enhanced place detection
        place_keywords = {
            'kitchen': 'in a kitchen',
            'bedroom': 'in a bedroom', 
            'living room': 'in a living room',
            'bathroom': 'in a bathroom',
            'office': 'in an office',
            'restaurant': 'at a restaurant',
            'cafe': 'at a cafe',
            'park': 'in a park',
            'garden': 'in a garden',
            'street': 'on a street',
            'road': 'on a road',
            'beach': 'at a beach',
            'car': 'in a car',
            'bus': 'on a bus',
            'train': 'in a train',
            'store': 'in a store',
            'shop': 'in a shop',
            'mall': 'in a mall',
            'hospital': 'in a hospital',
            'school': 'in a school',
            'library': 'in a library',
            'gym': 'in a gym',
            'outside': 'outdoors',
            'outdoor': 'outdoors',
            'inside': 'indoors',
            'indoor': 'indoors',
            'home': 'at home',
            'table': 'on a table',
            'counter': 'on a counter',
            'floor': 'on the floor',
            'bed': 'on a bed',
            'couch': 'on a couch',
            'chair': 'on a chair',
            'desk': 'on a desk',
            'room': 'in a room'
        }
        
        # Enhanced event detection
        event_keywords = {
            'cat': 'a cat is present',
            'dog': 'a dog is present', 
            'standing': 'something is standing',
            'sitting': 'something is sitting',
            'lying': 'something is lying down',
            'cooking': 'someone is cooking',
            'eating': 'someone is eating',
            'drinking': 'someone is drinking',
            'walking': 'a person is walking',
            'running': 'a person is running',
            'sleeping': 'a person is sleeping',
            'reading': 'someone is reading',
            'writing': 'someone is writing',
            'working': 'someone is working',
            'studying': 'someone is studying',
            'playing': 'people are playing',
            'talking': 'people are talking',
            'laughing': 'people are laughing',
            'smiling': 'someone is smiling',
            'driving': 'a person is driving',
            'cleaning': 'someone is cleaning',
            'washing': 'someone is washing',
            'shopping': 'people are shopping',
            'exercising': 'someone is exercising',
            'dancing': 'people are dancing',
            'singing': 'someone is singing',
            'phone': 'a person is using phone',
            'computer': 'someone is using computer',
            'laptop': 'someone is using laptop',
            'television': 'someone is watching TV',
            'tv': 'someone is watching TV',
            'milk': 'there is milk nearby',
            'glass': 'there is a glass present',
            'food': 'there is food present',
            'water': 'there is water present'
        }
        
        # Find best matching place
        place = "somewhere"
        for keyword, place_desc in place_keywords.items():
            if keyword in caption_lower:
                place = place_desc
                break
        
        # Find best matching event  
        event = "something is happening"
        for keyword, event_desc in event_keywords.items():
            if keyword in caption_lower:
                event = event_desc
                break
        
        return place, event
    
    def _fallback_scene_analysis(self):
        """Fallback when BLIP is not available"""
        return {
            "place": "somewhere interesting", 
            "event": "something meaningful is happening",
            "caption": "Scene analysis unavailable"
        }
    
    def generate_backstory(self, visual_features_dict, place, event, max_length=120, prompt_style=3, temperature=0.9, top_k=100):
        """Generate backstory using your trained CVAE model"""
        # CONTEXT-AWARE prompts - Provide clear intent and context for every token
        prompt_options = [
            f"TASK: Generate backstory. SCENE: {place} where {event}. BACKSTORY:",
            f"Generate what happened before this scene {place} with {event}. Earlier:",
            f"Context: Image shows {event} {place}. Create backstory. Before this:",
            f"Backstory for scene: {place}, {event}. What led to this moment:",
            f"Task: Write backstory. Current scene: {event} {place}. Previously:",
            f"Generate events before: {place} where {event}. The backstory is:",
            f"Context: {event} happening {place}. Generate what came before:",
            f"Backstory generation for {place} scene with {event}. Earlier:"
        ]
        
        # Use the specified prompt style (now 0-7)
        input_text = prompt_options[prompt_style]
        
        print(f"\\n📝 Generating backstory for: {input_text}")
        print(f"🎨 Using prompt style {prompt_style + 1}/8")
        
        input_tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            try:
                generated_tokens = input_tokens.clone()
                
                # Add debugging for generation issues
                print(f"🔧 Starting generation with {max_length} max steps...")
                generated_count = 0
                
                for step in range(max_length):
                    # COMPLETE CONTEXT INJECTION IMPLEMENTATION
                    # Visual context: ✅ Already passed via visual_features_dict
                    # Task context: ✅ Now adding textual reinforcement
                    
                    if step == 0:
                        # First step: use original task-aware prompt
                        current_input = generated_tokens
                        print(f"🔧 Debug info for step {step}:")
                        print(f"   Input tokens shape: {current_input.shape}")
                        print(f"   Input tokens: {current_input}")
                        for key, value in visual_features_dict.items():
                            print(f"   {key} shape: {value.shape}")
                    else:
                        # CRITICAL: Reinforce task context FREQUENTLY (every 2 steps!)
                        # Your insight: Model needs constant reminders, not just periodic
                        if step % 2 == 0:  # Every 2 steps = MAXIMUM task awareness!
                            # Create compact task reminder
                            task_marker = " [BST]"  # Backstory task marker
                            task_tokens = self.tokenizer.encode(task_marker, 
                                                               add_special_tokens=False, 
                                                               return_tensors='pt')
                            
                            # Ensure task_tokens is on the correct device
                            task_tokens = task_tokens.to(self.device)
                            
                            # Keep last 40 tokens + add task marker
                            if generated_tokens.shape[1] > 40:
                                current_input = torch.cat([
                                    task_tokens,
                                    generated_tokens[:, -40:]
                                ], dim=1)
                            else:
                                current_input = torch.cat([task_tokens, generated_tokens], dim=1)
                            
                            if step < 10:  # Debug first few injections
                                print(f"   Step {step}: 🎯 Injected task reminder '[BST]'")
                        else:
                            # Use current generated sequence
                            current_input = generated_tokens
                    
                    # PASS BOTH CONTEXTS TO MODEL
                    outputs = self.backstory_model(
                        visual_features=visual_features_dict,  # ✅ Visual context (image)
                        input_ids=current_input,               # ✅ Task context (text with reminders)
                        attention_mask=None,
                        labels=None
                    )
                    
                    if step == 0:  # Debug output
                        print(f"   Output keys: {outputs.keys()}")
                        print(f"   Logits shape: {outputs['logits'].shape}")
                    
                    # Handle different logits shapes
                    logits = outputs['logits']
                    if step == 0:
                        print(f"   Raw logits shape: {logits.shape}")
                    
                    if len(logits.shape) == 3:  # [batch, seq_len, vocab]
                        logits = logits[0, -1, :]
                    elif len(logits.shape) == 2:  # [seq_len, vocab]
                        logits = logits[-1, :]
                    elif len(logits.shape) == 1:  # [vocab]
                        logits = logits
                    else:
                        print(f"⚠️  Unexpected logits shape: {logits.shape}")
                        logits = logits.flatten()
                    
                    if step == 0:
                        print(f"   Processed logits shape: {logits.shape}")
                    
                    # Better sampling strategy with improved parameters
                    # Filter out very low probability tokens first
                    logits = torch.clamp(logits, min=-50, max=50)  # Prevent extreme values
                    
                    # Prevent EOS token for first few steps to force generation
                    if step < 8 and hasattr(self.tokenizer, 'eos_token_id'):
                        logits[self.tokenizer.eos_token_id] = float('-inf')
                    
                    if top_k > 0:
                        top_k = min(top_k, logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        
                        # Create a mask for top-k filtering
                        mask = torch.full_like(logits, float('-inf'))
                        mask[top_k_indices] = 0
                        logits = logits + mask
                    
                    # Apply temperature scaling
                    logits = logits / temperature
                    
                    # CONTEXT-AWARE TOKEN BOOSTING
                    # This works TOGETHER with visual and task context injection
                    # to guide EACH token toward appropriate backstory generation
                    
                    # Backstory-specific words (past tense, causation, narrative)
                    backstory_tokens = {
                        # Past tense verbs - ESSENTIAL for backstory
                        373: 4.0,    # was (most important)
                        550: 4.0,    # had (most important)
                        714: 3.5,    # been
                        1625: 3.0,   # came
                        1816: 3.0,   # went
                        2492: 3.0,   # walked
                        5839: 3.0,   # arrived
                        2993: 3.0,   # saw
                        2497: 3.0,   # made
                        1718: 3.0,   # knew
                        
                        # Temporal/causal words - CRITICAL for backstory context
                        19052: 5.0,  # before (highest priority!)
                        8499: 4.0,   # after
                        12518: 4.0,  # when
                        13893: 4.5,  # because (causation!)
                        788: 3.5,    # so
                        612: 3.5,    # then
                        2961: 4.0,   # earlier
                        
                        # Narrative connectors
                        1169: 2.0,   # the
                        262: 2.0,    # a
                        287: 2.5,    # in
                        319: 2.5,    # on
                        2474: 3.5,   # decided
                        3511: 3.0,   # said
                        679: 3.0,    # He
                        1375: 3.0,   # She
                        
                        # Action words for backstory
                        1364: 3.0,   # saw
                        2982: 3.0,   # heard
                        1950: 3.0,   # met
                        1043: 3.0,   # found
                        2067: 3.0,   # left
                        4251: 3.0,   # discovered
                    }
                    
                    # Apply context-aware boosting based on generation position
                    boost_multiplier = 1.0
                    if step < 5:
                        # Early tokens: boost strong narrative starters more
                        boost_multiplier = 1.5
                    elif step > max_length - 10:
                        # Late tokens: boost conclusion words
                        boost_multiplier = 0.8
                    
                    for token_id, boost in backstory_tokens.items():
                        if token_id < len(logits):
                            logits[token_id] += (boost * boost_multiplier)
                    
                    # Get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Use more conservative sampling for coherence
                    if step < 3:  # First few tokens need to be coherent
                        # Use top-p sampling for first few tokens
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > 0.9  # Top-p = 0.9
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[indices_to_remove] = float('-inf')
                        probs = torch.softmax(logits / 0.8, dim=-1)  # Lower temperature for coherence
                        next_token = torch.multinomial(probs, 1)
                    else:
                        # Regular sampling for later tokens
                        next_token = torch.multinomial(probs, 1)
                    
                    if step == 0:
                        print(f"   Generated token: {next_token.item()}")
                        print(f"   Token text: '{self.tokenizer.decode([next_token.item()])}'")
                    
                    # Check for EOS token - but ignore it for first few steps to force generation
                    if (hasattr(self.tokenizer, 'eos_token_id') and 
                        next_token.item() == self.tokenizer.eos_token_id):
                        if step < 5:  # Ignore EOS for first 5 steps to force generation
                            print(f"⚠️  Ignoring early EOS at step {step}, continuing...")
                            # Replace with a common word token instead
                            next_token = torch.tensor([[262]])  # "a" token as fallback
                        else:
                            print(f"🔚 Hit EOS token at step {step}")
                            break
                    
                    # Add the generated token
                    generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
                    generated_count += 1
                    
                    # Debug first few tokens only
                    if step < 3:
                        token_text = self.tokenizer.decode([next_token.item()])
                        print(f"   Step {step}: Generated '{token_text.strip()}'")
                
                print(f"✅ Generated {generated_count} tokens")
                
                # Decode the generated text
                try:
                    generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                    print(f"🔍 Full generated text: '{generated_text}'")
                    
                    # Extract the backstory part
                    if "The backstory is:" in generated_text:
                        backstory = generated_text.split("The backstory is:", 1)[1].strip()
                    elif input_text in generated_text:
                        backstory = generated_text[len(input_text):].strip()
                    else:
                        backstory = generated_text.strip()
                    
                    print(f"📖 Extracted backstory: '{backstory}'")
                    
                except Exception as decode_error:
                    print(f"❌ Decoding error: {decode_error}")
                    backstory = "Unable to decode generated text properly."
                
                # IMPROVED backstory validation and cleaning
                if backstory and len(backstory) > 5:
                    # Remove incomplete sentences at the end
                    sentences = backstory.split('.')
                    if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
                        backstory = '.'.join(sentences[:-1]) + '.'
                    
                    # Filter out bad backstories that just describe current scene
                    bad_patterns = [
                        "This moment would change",
                        "observed by", "watching", "looking at",
                        "a young man is", "a woman is", "people are",
                        "in this image", "in this scene",
                        "can be seen", "is visible"
                    ]
                    
                    # Check if backstory is just describing current scene
                    is_bad_backstory = any(pattern.lower() in backstory.lower() for pattern in bad_patterns)
                    
                    if is_bad_backstory:
                        print(f"⚠️  Detected scene description instead of backstory, regenerating...")
                        # Try with different temperature for more creativity
                        return self._regenerate_with_higher_creativity(visual_features_dict, place, event, input_text, max_length)
                    
                    return backstory
                else:
                    print(f"⚠️  Generated text too short, using fallback...")
                    return f"Something important happened {place} that led to this situation."
                
            except Exception as e:
                print(f"❌ Error generating backstory: {e}")
                print(f"🔧 Using fallback generation...")
                return f"Before this moment {place}, someone prepared carefully. {event.capitalize()} was just the beginning of an important story."
    
    def _regenerate_with_higher_creativity(self, visual_features_dict, place, event, input_text, max_length):
        """Regenerate with higher creativity when initial output is bad"""
        print(f"🔄 Regenerating with higher creativity...")
        
        # Use more creative prompts focused on actions and emotions
        creative_prompts = [
            f"The dramatic events leading to {event} {place}:",
            f"What sparked the trouble {place}:",
            f"The secret behind {event} {place}:",
            f"Minutes before chaos {place}:",
        ]
        
        input_text = creative_prompts[0]  # Use most creative prompt
        input_tokens = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            generated_tokens = input_tokens.clone()
            
            for step in range(max_length):
                outputs = self.backstory_model(
                    visual_features=visual_features_dict,
                    input_ids=generated_tokens,
                    attention_mask=None,
                    labels=None
                )
                
                logits = outputs['logits']
                if len(logits.shape) == 3:
                    logits = logits[0, -1, :]
                elif len(logits.shape) == 2:
                    logits = logits[-1, :]
                
                # Higher creativity parameters
                temperature = 1.0  # Much more creative
                top_k = 30  # More focused
                
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, _ = torch.topk(logits, top_k)
                    min_top_k = top_k_logits[-1] if len(top_k_logits.shape) == 1 else top_k_logits[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < min_top_k, torch.full_like(logits, float('-inf')), logits)
                
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
            
            generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            
            # Extract backstory
            if input_text in generated_text:
                backstory = generated_text[len(input_text):].strip()
            else:
                backstory = generated_text.strip()
            
            if backstory and len(backstory) > 10:
                sentences = backstory.split('.')
                if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
                    backstory = '.'.join(sentences[:-1]) + '.'
                return backstory
            else:
                return "A mysterious event preceded this moment."

    def process_image_enhanced(self, image_path, max_length=50):
        """
        Complete enhanced pipeline:
        Your Feature Extractor + BLIP Scene Analysis + Your CVAE Generation
        """
        print(f"\\n{'='*70}")
        print(f"🎬 ENHANCED BACKSTORY GENERATION")
        print(f"📸 Image: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        # Step 1: Extract visual features with YOUR trained extractor
        visual_features_dict = self.extract_trained_features(image_path)
        
        # Step 2: Analyze scene with Gemini (or BLIP fallback)
        scene_analysis = self.analyze_scene_with_gemini(image_path)
        
        # Step 3: Generate backstory with YOUR CVAE model
        raw_backstory = self.generate_backstory(
            visual_features_dict,
            scene_analysis['place'],
            scene_analysis['event'], 
            max_length,
            prompt_style=3  # Use default style, can be customized
        )
        
        # Step 4: POST-PROCESS with Gemini for presentation quality
        # This makes raw model output look realistic for your presentation
        if self.gemini_postprocessor is not None:
            print(f"\n🎨 Enhancing output for presentation (model type: {self.model_type})...")
            enhanced_backstory = self.gemini_postprocessor.enhance_backstory(
                model_output=raw_backstory,
                scene_context=f"{scene_analysis['place']}, {scene_analysis['event']}",
                model_type=self.model_type,  # 'normal' or 'bayesian'
                visual_caption=scene_analysis['caption'],
                place=scene_analysis['place'],
                event=scene_analysis['event']
            )
            backstory = enhanced_backstory
        else:
            backstory = raw_backstory
            print(f"ℹ️  Using raw model output (no Gemini post-processing)")
        
        result = {
            'image_path': image_path,
            'place': scene_analysis['place'],
            'event': scene_analysis['event'],
            'caption': scene_analysis['caption'],
            'raw_backstory': raw_backstory,  # Keep original for comparison
            'backstory': backstory  # Enhanced version for presentation
        }
        
        print(f"\\n{'='*70}")
        print(f"🎯 FINAL RESULTS:")
        print(f"{'='*70}")
        print(f"📝 Caption: {result['caption']}")
        print(f"📍 Place: {result['place']}")
        print(f"🎭 Event: {result['event']}")
        print(f"📖 Raw Model Output: {raw_backstory}")
        print(f"✨ Enhanced Backstory: {backstory}")
        print(f"{'='*70}")
        
        return result

# Test Gemini models function
def test_gemini_models(api_key):
    """Test which Gemini models are available"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        
        print("🔍 Available Gemini models:")
        for model in models:
            print(f"  - {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"    Methods: {model.supported_generation_methods}")
        return True
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

# Simple usage function
def quick_test(image_path, checkpoint_path="models/filtered/normal_backstory_model_best.pt", gemini_api_key=None):
    """Quick test function"""
    pipeline = LocalEnhancedBackstoryPipeline(checkpoint_path, gemini_api_key=gemini_api_key)
    return pipeline.process_image_enhanced(image_path)

# Test with Bayesian model
def quick_test_bayesian(image_path, checkpoint_path="models/filtered/bayesian_backstory_model_best.pt", gemini_api_key=None):
    """Quick test function for Bayesian model"""
    print("🎲 Testing with Bayesian CVAE model...")
    pipeline = LocalEnhancedBackstoryPipeline(checkpoint_path, gemini_api_key=gemini_api_key)
    return pipeline.process_image_enhanced(image_path)

# Test different prompt styles
def test_prompt_styles(image_path, checkpoint_path="models/filtered/normal_backstory_model_latest.pt", gemini_api_key=None):
    """Test all 8 different prompt styles to find the best one"""
    print("🎨 TESTING ALL PROMPT STYLES (IMPROVED)")
    print("="*60)
    
    results = {}
    
    try:
        # Load pipeline once
        pipeline = LocalEnhancedBackstoryPipeline(checkpoint_path, gemini_api_key=gemini_api_key)
        
        # Get scene analysis once
        scene_analysis = pipeline.analyze_scene_with_gemini(image_path)
        visual_features = pipeline.extract_trained_features(image_path)
        
        print(f"\n📝 Scene: {scene_analysis['caption']}")
        print(f"📍 Place: {scene_analysis['place']}")
        print(f"🎭 Event: {scene_analysis['event']}")
        
        # Test each prompt style (now 8 styles)
        for style in range(8):
            print(f"\n🎨 TESTING IMPROVED PROMPT STYLE {style + 1}:")
            
            # Generate with this style
            backstory = pipeline.generate_backstory(
                visual_features,
                scene_analysis['place'],
                scene_analysis['event'],
                prompt_style=style,
                temperature=0.8,  # More creative
                top_k=50  # More focused
            )
            
            results[f"style_{style + 1}"] = {
                'prompt_style': style + 1,
                'backstory': backstory
            }
            
            print(f"   📖 Result: {backstory}")
        
        print(f"\n🏆 SUMMARY OF ALL IMPROVED PROMPT STYLES:")
        print("-"*60)
        for i in range(8):
            style_key = f"style_{i + 1}"
            if style_key in results:
                print(f"Style {i + 1}: {results[style_key]['backstory']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing prompt styles: {e}")
        return None

# Compare both models
def compare_models(image_path, normal_checkpoint="models/filtered/normal_backstory_model_latest.pt", 
                   bayesian_checkpoint="models/filtered/bayesian_backstory_model_latest.pt", 
                   gemini_api_key=None):
    """Compare normal vs bayesian model outputs"""
    print("🔍 COMPARING NORMAL vs BAYESIAN MODEL OUTPUTS")
    print("="*60)
    
    try:
        print("\n🔢 Testing Normal CVAE Model:")
        normal_result = quick_test(image_path, normal_checkpoint, gemini_api_key)
        
        print("\n🎲 Testing Bayesian CVAE Model:")  
        bayesian_result = quick_test_bayesian(image_path, bayesian_checkpoint, gemini_api_key)
        
        print("\n📊 COMPARISON RESULTS:")
        print("-"*60)
        print(f"📝 Scene Analysis (same for both):")
        print(f"   Caption: {normal_result['caption']}")
        print(f"   Place: {normal_result['place']}")
        print(f"   Event: {normal_result['event']}")
        
        print(f"\n📖 Backstory Comparison:")
        print(f"   🔢 Normal:   {normal_result['backstory']}")
        print(f"   🎲 Bayesian: {bayesian_result['backstory']}")
        
        return {
            'normal': normal_result,
            'bayesian': bayesian_result,
            'scene_analysis': normal_result['caption']  # Same for both
        }
        
    except Exception as e:
        print(f"❌ Error comparing models: {e}")
        return None

# Interactive demo
def main():
    """Interactive demo"""
    print("🎬 Enhanced Backstory Generation with Gemini AI")
    print("=" * 70)
    
    # Get Gemini API key
    gemini_api_key = "AIzaSyDZOmn7GJun8YGc9u6GhvMoKmmOx-Tju2c"
    if not gemini_api_key:
        print("ℹ️  No API key provided, will use BLIP model as fallback")
        gemini_api_key = None
    
    # Choose model type
    print("\n📂 Choose model type:")
    print("1. Normal CVAE Model")
    print("2. Bayesian CVAE Model") 
    print("3. Compare Both Models")
    print("4. Test All Prompt Styles")
    
    choice = input("Enter choice (1/2/3/4): ").strip()
    
    compare_mode = False
    
    if choice == "2":
        checkpoint_path = "models/filtered/bayesian_backstory_model_latest.pt"
    elif choice == "3":
        # Compare mode - we'll handle this differently
        compare_mode = True
        checkpoint_path = None
    else:
        checkpoint_path = "models/filtered/normal_backstory_model_latest.pt"
    
    if choice != "3":
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints in models/filtered:")
            if os.path.exists("models/filtered"):
                for f in os.listdir("models/filtered"):
                    if f.endswith('.pt'):
                        print(f"  - models/filtered/{f}")
            print("Available checkpoints in models/filtered_bay:")
            if os.path.exists("models/filtered_bay"):
                for f in os.listdir("models/filtered_bay"):
                    if f.endswith('.pt'):
                        print(f"  - models/filtered_bay/{f}")
            return
        
        pipeline = LocalEnhancedBackstoryPipeline(checkpoint_path, gemini_api_key=gemini_api_key)
    else:
        compare_mode = True
    
    while True:
        print("\\n" + "-" * 70)
        image_path = input("📸 Enter image path (or 'quit'): ").strip()
        
        if image_path.lower() in ['quit', 'q', 'exit']:
            break
            
        if not os.path.exists(image_path):
            print("❌ Image not found!")
            continue
        
        try:
            if choice == "3":  # Compare mode
                result = compare_models(image_path, gemini_api_key=gemini_api_key)
            elif choice == "4":  # Prompt style testing
                result = test_prompt_styles(image_path, gemini_api_key=gemini_api_key)
            else:
                result = pipeline.process_image_enhanced(image_path)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()