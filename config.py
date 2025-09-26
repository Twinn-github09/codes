import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
VCR_IMAGES_DIR = r'g:\projects\DL-PR\vcr1images'  # Update this to your VCR images location
if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")

VCR_FEATURES_DIR = r'G:\projects\DL-PR\visual-comet\data\visualcomet'