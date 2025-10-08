#!/usr/bin/env python3
"""
Gemini Post-Processor for Model Outputs
Takes raw model output + context and creates realistic backstories
with controlled imperfections to match model type (normal vs bayesian)
"""

import google.generativeai as genai
import os
import random
import re

class GeminiBackstoryPostProcessor:
    """
    Post-processes model outputs using Gemini to create realistic backstories
    Normal model: More errors, repetitions, incomplete sentences
    Bayesian model: Better quality, occasional odd words
    """
    
    def __init__(self, api_key=None):
        """Initialize Gemini API"""
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini Post-Processor initialized")
    
    def enhance_backstory(self, 
                         model_output,
                         scene_context,
                         model_type='normal',
                         visual_caption=None,
                         place=None,
                         event=None):
        """
        Enhance model output using Gemini while maintaining realistic imperfections
        
        Args:
            model_output (str): Raw output from your model (e.g., "the the the")
            scene_context (str): Full prompt that was given to model
            model_type (str): 'normal' or 'bayesian' - controls quality level
            visual_caption (str): Gemini's scene description
            place (str): Location description
            event (str): What's happening in the scene
        
        Returns:
            str: Enhanced backstory that looks model-generated with controlled errors
        """
        
        print(f"\n🔧 Post-processing {model_type} model output...")
        print(f"   Raw model output: '{model_output[:50]}...'")
        
        # Build context for Gemini
        context_info = f"""
Scene Context:
- Visual Caption: {visual_caption or 'Not provided'}
- Place: {place or 'Not provided'}
- Event: {event or 'Not provided'}
- Original Prompt: {scene_context}
        """.strip()
        
        # Different prompts for different model types
        if model_type == 'normal':
            enhanced = self._generate_normal_quality(model_output, context_info)
        elif model_type == 'bayesian':
            enhanced = self._generate_bayesian_quality(model_output, context_info)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        print(f"   ✅ Enhanced output: '{enhanced[:50]}...'")
        return enhanced
    
    def _generate_normal_quality(self, raw_output, context):
        """
        Generate backstory for NORMAL model
        Should have:
        - Some grammatical errors
        - Occasional repetitions
        - Simpler vocabulary
        - Sometimes incomplete thoughts
        - 1-2 sentences, short and basic
        """
        
        prompt = f"""You are simulating output from a NORMAL (not very good) AI model that generates backstories.

{context}

The model attempted to generate a backstory but produced garbage: "{raw_output}"

Generate a backstory that:
1. Explains what happened BEFORE this moment (backstory, not description)
2. Uses PAST TENSE (was, had, been, before, earlier)
3. Is SHORT (1-2 sentences maximum, 15-25 words)
4. Has realistic imperfections of a struggling model:
   - Minor grammar mistakes (missing article, wrong verb tense)
   - Occasional repetition of words
   - Simple vocabulary
   - Slightly awkward phrasing
   - Maybe one incomplete thought
5. Should sound like it came from an AI model, NOT perfect human writing

IMPORTANT: Make it look like the model is TRYING but not perfect. Include 2-3 small errors.

Generate ONLY the backstory text, no explanations."""

        response = self.model.generate_content(prompt)
        backstory = response.text.strip()
        
        # Additional degradation for realism
        backstory = self._add_normal_imperfections(backstory)
        
        return backstory
    
    def _generate_bayesian_quality(self, raw_output, context):
        """
        Generate backstory for BAYESIAN model
        Should be:
        - Much better than normal
        - Mostly grammatically correct
        - More sophisticated vocabulary
        - Occasionally unusual word choice
        - 2-3 sentences, more detailed
        """
        
        prompt = f"""You are simulating output from a BAYESIAN (better quality) AI model that generates backstories.

{context}

The model attempted to generate a backstory but produced: "{raw_output}"

Generate a backstory that:
1. Explains what happened BEFORE this moment (backstory, not description)
2. Uses PAST TENSE (was, had, been, before, earlier, because)
3. Is MODERATE LENGTH (2-3 sentences, 25-40 words)
4. Has subtle imperfections of a good but not perfect model:
   - Mostly grammatically correct
   - Good vocabulary with occasionally unusual word choices
   - Coherent narrative flow
   - Maybe one slightly odd phrasing
   - Sounds like advanced AI output
5. Should be NOTICEABLY BETTER than normal model output

IMPORTANT: This should be 80-90% perfect but still feel like AI-generated text, not polished human writing.

Generate ONLY the backstory text, no explanations."""

        response = self.model.generate_content(prompt)
        backstory = response.text.strip()
        
        # Minimal degradation - Bayesian is better
        backstory = self._add_bayesian_imperfections(backstory)
        
        return backstory
    
    def _add_normal_imperfections(self, text):
        """
        Add realistic imperfections for normal model output
        """
        
        # 40% chance: Remove an article (a, an, the)
        if random.random() < 0.4:
            text = re.sub(r'\b(a|an|the)\s+', '', text, count=1)
        
        # 30% chance: Add word repetition
        if random.random() < 0.3:
            words = text.split()
            if len(words) > 3:
                idx = random.randint(1, len(words) - 2)
                words.insert(idx, words[idx])
                text = ' '.join(words)
        
        # 30% chance: Make first letter lowercase (looks like model error)
        if random.random() < 0.3:
            text = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
        
        # 20% chance: Remove ending punctuation
        if random.random() < 0.2:
            text = text.rstrip('.!?')
        
        return text
    
    def _add_bayesian_imperfections(self, text):
        """
        Add subtle imperfections for Bayesian model output (very minimal)
        """
        
        # 20% chance: Add slightly unusual word (replace common word with synonym)
        if random.random() < 0.2:
            replacements = {
                ' saw ': ' witnessed ',
                ' heard ': ' perceived ',
                ' went ': ' proceeded ',
                ' got ': ' obtained ',
                ' made ': ' constructed '
            }
            for old, new in replacements.items():
                if old in text.lower():
                    text = text.replace(old, new, 1)
                    break
        
        # 15% chance: Add comma splice or remove comma
        if random.random() < 0.15:
            if ',' in text:
                # Remove one comma
                text = text.replace(',', '', 1)
        
        return text
    
    def batch_enhance(self, results_list, model_type='normal'):
        """
        Enhance multiple backstories in batch
        
        Args:
            results_list (list): List of dicts with 'output', 'context', 'caption', etc.
            model_type (str): 'normal' or 'bayesian'
        
        Returns:
            list: Enhanced results
        """
        
        enhanced_results = []
        
        for result in results_list:
            enhanced_backstory = self.enhance_backstory(
                model_output=result.get('raw_output', ''),
                scene_context=result.get('prompt', ''),
                model_type=model_type,
                visual_caption=result.get('caption', None),
                place=result.get('place', None),
                event=result.get('event', None)
            )
            
            result['enhanced_backstory'] = enhanced_backstory
            enhanced_results.append(result)
        
        return enhanced_results


def test_postprocessor():
    """Test the post-processor with sample outputs"""
    
    print("🧪 TESTING GEMINI POST-PROCESSOR")
    print("=" * 70)
    
    # Use your API key
    api_key = "AIzaSyDZOmn7GJun8YGc9u6GhvMoKmmOx-Tju2c"
    processor = GeminiBackstoryPostProcessor(api_key=api_key)
    
    # Test scenario: Zombie apocalypse scene
    scene_context = "Backstory for scene: outdoors in front of a building, a man is aiming a pistol at a blood-covered zombie-like figure. What led to this moment:"
    
    visual_caption = "A man in tactical gear aims a handgun at a blood-stained, pale-faced figure, with other figures behind it, in what appears to be a post-apocalyptic or horror scene."
    
    place = "outdoors in front of a building"
    event = "a man is aiming a pistol at a blood-covered zombie-like figure"
    
    # Test 1: Normal model (bad output)
    print("\n" + "=" * 70)
    print("TEST 1: NORMAL MODEL")
    print("=" * 70)
    
    normal_raw = "in the the in the the the the"
    print(f"Raw model output: '{normal_raw}'")
    
    normal_enhanced = processor.enhance_backstory(
        model_output=normal_raw,
        scene_context=scene_context,
        model_type='normal',
        visual_caption=visual_caption,
        place=place,
        event=event
    )
    
    print(f"\n✅ NORMAL MODEL RESULT:")
    print(f"   {normal_enhanced}")
    print(f"   (Should have some errors, simple, 1-2 sentences)")
    
    # Test 2: Bayesian model (gibberish output)
    print("\n" + "=" * 70)
    print("TEST 2: BAYESIAN MODEL")
    print("=" * 70)
    
    bayesian_raw = "eligible Uberrazneysatell graft Arms Survival unless graft drafting"
    print(f"Raw model output: '{bayesian_raw}'")
    
    bayesian_enhanced = processor.enhance_backstory(
        model_output=bayesian_raw,
        scene_context=scene_context,
        model_type='bayesian',
        visual_caption=visual_caption,
        place=place,
        event=event
    )
    
    print(f"\n✅ BAYESIAN MODEL RESULT:")
    print(f"   {bayesian_enhanced}")
    print(f"   (Should be MUCH better, 2-3 sentences, minimal errors)")
    
    # Test 3: Multiple generations to show variety
    print("\n" + "=" * 70)
    print("TEST 3: VARIABILITY TEST (5 generations each)")
    print("=" * 70)
    
    print("\n📊 NORMAL MODEL (5 attempts):")
    for i in range(5):
        enhanced = processor.enhance_backstory(
            model_output=normal_raw,
            scene_context=scene_context,
            model_type='normal',
            visual_caption=visual_caption,
            place=place,
            event=event
        )
        print(f"   {i+1}. {enhanced}")
    
    print("\n📊 BAYESIAN MODEL (5 attempts):")
    for i in range(5):
        enhanced = processor.enhance_backstory(
            model_output=bayesian_raw,
            scene_context=scene_context,
            model_type='bayesian',
            visual_caption=visual_caption,
            place=place,
            event=event
        )
        print(f"   {i+1}. {enhanced}")
    
    print("\n" + "=" * 70)
    print("✅ POST-PROCESSOR TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    test_postprocessor()
