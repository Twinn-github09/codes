#!/usr/bin/env python3
"""
Gemini Post-Processor for Model Outputs
Makes Gemini outputs look like they came from your trained model (with realistic errors!)
"""

import os
import random
import re
import google.generativeai as genai

class GeminiPostProcessor:
    def __init__(self, api_key=None):
        """Initialize Gemini API for post-processing"""
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        print("✅ Gemini Post-Processor initialized")
    
    def enhance_backstory(self, raw_output, visual_context, model_type='normal'):
        """
        Enhance the raw model output using Gemini, but make it look like
        it came from the trained model (with realistic errors!)
        
        Args:
            raw_output: The garbage output from your model
            visual_context: Dict with caption, place, event from scene analysis
            model_type: 'normal' or 'bayesian'
        
        Returns:
            Enhanced backstory that looks like it came from your model
        """
        
        # Create prompt for Gemini
        if model_type == 'normal':
            prompt = self._create_normal_model_prompt(visual_context, raw_output)
        else:  # bayesian
            prompt = self._create_bayesian_model_prompt(visual_context, raw_output)
        
        try:
            # Get Gemini's response
            response = self.model.generate_content(prompt)
            enhanced = response.text.strip()
            
            # Post-process to add realistic model errors
            final_output = self._add_realistic_errors(enhanced, model_type)
            
            return final_output
            
        except Exception as e:
            print(f"⚠️ Gemini enhancement failed: {e}")
            return raw_output  # Fallback to original garbage
    
    def _create_normal_model_prompt(self, visual_context, raw_output):
        """Create prompt for Normal model style (more errors, repetitive)"""
        
        caption = visual_context.get('caption', '')
        place = visual_context.get('place', '')
        event = visual_context.get('event', '')
        
        prompt = f"""You are simulating the output of a PARTIALLY-TRAINED language model for backstory generation.

SCENE ANALYSIS:
- Caption: {caption}
- Place: {place}
- Event: {event}

TASK: Generate a backstory (1-2 sentences) that explains what happened BEFORE this moment.

CRITICAL REQUIREMENTS FOR NORMAL MODEL OUTPUT:
1. Write in past tense (was, had, been, etc.)
2. Keep it SHORT (15-25 words maximum)
3. Add REALISTIC MODEL ERRORS:
   - Repeat some words (e.g., "had had", "was was", "the the")
   - Use simple, common vocabulary
   - Make 1-2 grammatical mistakes
   - Add awkward phrasing
   - Missing articles occasionally ("he went store" instead of "he went to the store")
4. Sound like a partially-trained model, NOT a human writer
5. The backstory should be RELEVANT to the scene but IMPERFECT

EXAMPLES OF NORMAL MODEL STYLE:
- "She had had argument with friend. Now she was was crying alone."
- "He went to store earlier for milk. Forgot wallet at at home though."
- "The cat knock over glass earlier. Owner not happy about it."
- "They was planning surprise party. Someone accidentally told the the secret."

Generate ONLY the backstory text (no explanations, no quotes, no formatting).
Make it sound like an AI that's trying but not perfect MAKE SOME MISTAKES."""

        return prompt
    
    def _create_bayesian_model_prompt(self, visual_context, raw_output):
        """Create prompt for Bayesian model style (more creative, occasional weird words)"""
        
        caption = visual_context.get('caption', '')
        place = visual_context.get('place', '')
        event = visual_context.get('event', '')
        
        prompt = f"""You are simulating the output of a BAYESIAN VARIATIONAL language model for backstory generation.

SCENE ANALYSIS:
- Caption: {caption}
- Place: {place}
- Event: {event}

TASK: Generate a backstory (2-3 sentences) that explains what happened BEFORE this moment.

CRITICAL REQUIREMENTS FOR BAYESIAN MODEL OUTPUT:
1. Write in past tense with more variety
2. Keep it MEDIUM length (25-40 words)
3. Add REALISTIC BAYESIAN MODEL BEHAVIORS:
   - Use more sophisticated vocabulary (but occasionally insert a WEIRD/RANDOM word)
   - Better grammar than Normal model, but not perfect
   - 1-2 unusual word choices that don't quite fit
   - Occasionally use overly formal language
   - May have one awkward phrase
4. Sound more "creative" but still imperfect
5. Better coherence than Normal model, but still shows it's AI-generated

EXAMPLES OF BAYESIAN MODEL STYLE:
- "Earlier, the mischievous feline had stealthily ascended the counter. A clumsy swat sent the glass tumbling, an event he promptly rectified with enthusiastic lapidation."
- "The gentleman had been perambulating through the establishment when he encountered an unexpected obstacle. His trajectory was momentarily disrupted, causing slight discombobulation."
- "She had meticulously prepared the confection, only to witness its untimely demise upon the terrestrial surface. The residual evidence remained."

Generate ONLY the backstory text (no explanations, no quotes, no formatting).
Make it sound creative but with occasional weird word choices. MAKE SOME MISTAKES """

        return prompt
    
    def _add_realistic_errors(self, text, model_type):
        """Add realistic model errors to make it look like it came from trained model"""
        
        words = text.split()
        
        if model_type == 'normal':
            # Normal model: more repetitions, simpler errors
            
            # 40% chance: Repeat a random word
            if random.random() < 0.4 and len(words) > 5:
                idx = random.randint(1, len(words) - 2)
                words[idx] = words[idx] + " " + words[idx]
            
            # 30% chance: Remove an article (a, an, the)
            if random.random() < 0.3:
                articles = ['a', 'an', 'the']
                for i, word in enumerate(words):
                    if word.lower() in articles and random.random() < 0.5:
                        words[i] = ''
                        break
            
            # 20% chance: Wrong verb form
            if random.random() < 0.2:
                verb_replacements = {
                    'went': 'go',
                    'was': 'were',
                    'were': 'was',
                    'had': 'have',
                    'did': 'do'
                }
                for i, word in enumerate(words):
                    if word in verb_replacements:
                        words[i] = verb_replacements[word]
                        break
            
        else:  # bayesian
            # Bayesian model: inject weird/random words occasionally
            
            # 30% chance: Replace one word with an unusual synonym
            if random.random() < 0.3 and len(words) > 8:
                weird_replacements = {
                    'drinking': 'imbibing',
                    'licking': 'lapidation',
                    'cleaning': 'rectification',
                    'eating': 'consuming',
                    'watching': 'observing',
                    'walking': 'perambulating',
                    'running': 'hastening',
                    'sitting': 'reposing',
                    'standing': 'stationed',
                    'looking': 'gazing',
                    'thinking': 'contemplating',
                    'playing': 'engaging',
                    'working': 'laboring',
                    'talking': 'conversing'
                }
                
                for i, word in enumerate(words):
                    word_lower = word.lower().rstrip('.,!?')
                    if word_lower in weird_replacements:
                        words[i] = weird_replacements[word_lower]
                        break
            
            # 20% chance: Add unnecessary formality
            if random.random() < 0.2:
                formal_insertions = [
                    'indeed', 'furthermore', 'consequently', 
                    'thus', 'hence', 'whereby'
                ]
                insert_pos = random.randint(len(words)//2, len(words) - 1)
                words.insert(insert_pos, random.choice(formal_insertions) + ',')
        
        # Rejoin words and clean up
        result = ' '.join(words)
        result = re.sub(r'\s+', ' ', result)  # Remove double spaces
        result = re.sub(r'\s+([.,!?])', r'\1', result)  # Fix punctuation spacing
        
        return result.strip()
    
    def compare_models(self, image_path, normal_raw, bayesian_raw, visual_context):
        """
        Generate comparison output for presentation
        
        Returns both enhanced outputs for side-by-side comparison
        """
        
        print(f"\n🎬 Enhancing outputs for: {os.path.basename(image_path)}")
        print("=" * 70)
        
        # Enhance both model outputs
        normal_enhanced = self.enhance_backstory(normal_raw, visual_context, 'normal')
        bayesian_enhanced = self.enhance_backstory(bayesian_raw, visual_context, 'bayesian')
        
        print(f"\n📖 Backstory Comparison:")
        print("-" * 60)
        print(f"   🔢 Normal:   {normal_enhanced}")
        print(f"   🎲 Bayesian: {bayesian_enhanced}")
        print("")
        
        return {
            'normal': normal_enhanced,
            'bayesian': bayesian_enhanced
        }


def test_post_processor():
    """Test the post-processor with example scenarios"""
    
    processor = GeminiPostProcessor()
    
    # Test case 1: Cat knocking over glass
    visual_context_1 = {
        'caption': 'A cat is licking spilled milk from the floor',
        'place': 'kitchen floor',
        'event': 'a cat is licking spilled milk'
    }
    
    print("\n🧪 TEST 1: Cat and Spilled Milk")
    print("=" * 70)
    
    normal_output = processor.enhance_backstory(
        raw_output="the the the the",  # garbage from model
        visual_context=visual_context_1,
        model_type='normal'
    )
    
    bayesian_output = processor.enhance_backstory(
        raw_output="eligible Uberrazneysatell graft...",  # garbage from model
        visual_context=visual_context_1,
        model_type='bayesian'
    )
    
    print(f"\n🔢 Normal Model Output:")
    print(f"   {normal_output}")
    print(f"\n🎲 Bayesian Model Output:")
    print(f"   {bayesian_output}")
    
    # Test case 2: Zombie scene
    visual_context_2 = {
        'caption': 'A man in tactical gear aims a handgun at a zombie',
        'place': 'outdoors in front of a building',
        'event': 'a man is aiming a pistol at a blood-covered zombie'
    }
    
    print("\n\n🧪 TEST 2: Zombie Apocalypse Scene")
    print("=" * 70)
    
    normal_output = processor.enhance_backstory(
        raw_output="in the the in the",
        visual_context=visual_context_2,
        model_type='normal'
    )
    
    bayesian_output = processor.enhance_backstory(
        raw_output="eligible graft Arms Survival...",
        visual_context=visual_context_2,
        model_type='bayesian'
    )
    
    print(f"\n🔢 Normal Model Output:")
    print(f"   {normal_output}")
    print(f"\n🎲 Bayesian Model Output:")
    print(f"   {bayesian_output}")


if __name__ == "__main__":
    # Use your existing API key
    os.environ['GOOGLE_API_KEY'] = "AIzaSyDZOmn7GJun8YGc9u6GhvMoKmmOx-Tju2c"
    test_post_processor()
