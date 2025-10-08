"""
Manual backstory generation with user-specified place and event
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.image_to_backstory import ImageToBackstoryPipeline

def interactive_backstory_generation():
    """Interactive backstory generation with manual place/event input"""
    
    pipeline = ImageToBackstoryPipeline('models/filtered/normal_backstory_model_latest.pt')
    
    print("🎬 Manual Backstory Generation")
    print("=" * 50)
    
    while True:
        print("\\n" + "-" * 50)
        
        # Get image path
        image_path = input("📸 Enter image path (or 'quit'): ").strip()
        if image_path.lower() in ['quit', 'q', 'exit']:
            break
            
        if not os.path.exists(image_path):
            print("❌ Image not found!")
            continue
        
        print("\\nChoose input method:")
        print("1. Auto-detect place and event (CLIP-based)")
        print("2. Manually specify place and event (more accurate)")
        print("3. Try multiple place/event combinations")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            # Auto-detection
            result = pipeline.process_image(image_path)
            print(f"\\n🎯 AUTO-DETECTED RESULT:")
            print(f"📍 Place: {result['place']}")
            print(f"🎭 Event: {result['event']}")
            print(f"📖 Backstory: {result['backstory']}")
            
        elif choice == "2":
            # Manual input
            print("\\n📍 PLACE OPTIONS:")
            places = [
                "in a kitchen", "in a living room", "in a bedroom", "in a bathroom",
                "outdoors", "in a park", "on a street", "in a car", "on a bus",
                "at a restaurant", "in a store", "in an office", "at home",
                "in a school", "in a hospital", "at a beach"
            ]
            for i, place in enumerate(places, 1):
                print(f"{i:2d}. {place}")
            
            place_choice = input("\\nEnter place number or custom place: ").strip()
            try:
                place_idx = int(place_choice) - 1
                if 0 <= place_idx < len(places):
                    place = places[place_idx]
                else:
                    raise ValueError
            except ValueError:
                place = place_choice if place_choice else "somewhere"
            
            print("\\n🎭 EVENT OPTIONS:")
            events = [
                "someone is cooking", "someone is eating", "someone is cleaning",
                "a person is walking", "a person is sitting", "a person is sleeping",
                "someone is working", "someone is reading", "someone is talking",
                "a person is driving", "people are shopping", "someone is exercising",
                "a person is using phone", "someone is watching TV", "people are playing"
            ]
            for i, event in enumerate(events, 1):
                print(f"{i:2d}. {event}")
            
            event_choice = input("\\nEnter event number or custom event: ").strip()
            try:
                event_idx = int(event_choice) - 1
                if 0 <= event_idx < len(events):
                    event = events[event_idx]
                else:
                    raise ValueError
            except ValueError:
                event = event_choice if event_choice else "something is happening"
            
            # Generate with manual inputs
            image_features = pipeline.extract_image_features(image_path)
            visual_features_dict = pipeline.prepare_visual_features_for_backstory_model(image_features)
            backstory = pipeline.generate_backstory(visual_features_dict, place, event)
            
            print(f"\\n🎯 MANUAL INPUT RESULT:")
            print(f"📍 Place: {place}")
            print(f"🎭 Event: {event}")
            print(f"📖 Backstory: {backstory}")
            
        elif choice == "3":
            # Multiple combinations
            print("\\n🎲 TRYING MULTIPLE COMBINATIONS:")
            
            combinations = [
                ("in a kitchen", "someone is cooking"),
                ("in a living room", "people are talking"),
                ("outdoors", "a person is walking"),
                ("at a restaurant", "someone is eating"),
                ("in an office", "someone is working")
            ]
            
            image_features = pipeline.extract_image_features(image_path)
            visual_features_dict = pipeline.prepare_visual_features_for_backstory_model(image_features)
            
            for i, (place, event) in enumerate(combinations, 1):
                backstory = pipeline.generate_backstory(visual_features_dict, place, event)
                print(f"\\n{i}. 📍 {place} + 🎭 {event}")
                print(f"   📖 {backstory}")

if __name__ == "__main__":
    interactive_backstory_generation()