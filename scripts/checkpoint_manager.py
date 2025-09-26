#!/usr/bin/env python3
"""
Checkpoint Manager for Visual Backstory Preprocessing
This script provides utilities to manage preprocessing checkpoints.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess_backstory import BackstoryPreprocessor

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str

def show_detailed_status(preprocessor):
    """Show detailed checkpoint status"""
    print("📊 Detailed Preprocessing Status")
    print("=" * 50)
    
    # Check progress checkpoint
    if os.path.exists(preprocessor.progress_checkpoint):
        with open(preprocessor.progress_checkpoint, 'r') as f:
            progress_data = json.load(f)
        
        print("🎯 Feature Extraction Progress:")
        print(f"   Last Updated: {format_timestamp(progress_data.get('timestamp', 'Unknown'))}")
        print(f"   Processed Images: {len(progress_data.get('processed_image_list', []))}")
        print(f"   Skipped (Existing): {progress_data.get('skipped_existing', 0)}")
        print(f"   Skipped (Missing): {progress_data.get('skipped_missing', 0)}")
        print(f"   Total Images: {progress_data.get('total_images', 0)}")
        
        if progress_data.get('total_images', 0) > 0:
            completed = len(progress_data.get('processed_image_list', [])) + progress_data.get('skipped_existing', 0) + progress_data.get('skipped_missing', 0)
            progress_percent = completed / progress_data['total_images'] * 100
            print(f"   Progress: {progress_percent:.1f}% complete")
        
        print(f"   Checkpoint File: {preprocessor.progress_checkpoint}")
    else:
        print("🎯 Feature Extraction Progress: No checkpoint found")
    
    print()
    
    # Check annotations checkpoint
    if os.path.exists(preprocessor.annotations_checkpoint):
        with open(preprocessor.annotations_checkpoint, 'r') as f:
            annotations_data = json.load(f)
        
        print("📋 Annotations Processing:")
        total_annotations = sum(len(split_data) for split_data in annotations_data.values())
        print(f"   Total Processed Annotations: {total_annotations}")
        
        for split, split_data in annotations_data.items():
            print(f"   {split.title()}: {len(split_data)} entries")
        
        print(f"   Checkpoint File: {preprocessor.annotations_checkpoint}")
    else:
        print("📋 Annotations Processing: No checkpoint found")
    
    print()
    
    # Check output directories
    print("📁 Output Status:")
    
    # Features directory
    if os.path.exists(preprocessor.features_dir):
        feature_files = [f for f in os.listdir(preprocessor.features_dir) if f.endswith('.pkl')]
        print(f"   Feature Files: {len(feature_files)} .pkl files")
    else:
        print("   Feature Files: Directory not found")
    
    # Processed directory
    if os.path.exists(preprocessor.processed_dir):
        processed_files = os.listdir(preprocessor.processed_dir)
        print(f"   Processed Files: {processed_files}")
    else:
        print("   Processed Files: Directory not found")

def list_checkpoint_files(preprocessor):
    """List all checkpoint files"""
    print("📋 Checkpoint Files:")
    print("=" * 30)
    
    if os.path.exists(preprocessor.checkpoint_dir):
        checkpoint_files = os.listdir(preprocessor.checkpoint_dir)
        
        if checkpoint_files:
            for file in checkpoint_files:
                file_path = os.path.join(preprocessor.checkpoint_dir, file)
                file_size = os.path.getsize(file_path)
                file_size_kb = file_size / 1024
                
                print(f"   {file} ({file_size_kb:.1f} KB)")
        else:
            print("   No checkpoint files found")
    else:
        print("   Checkpoint directory not found")

def backup_checkpoints(preprocessor, backup_dir):
    """Backup checkpoint files"""
    import shutil
    
    if not os.path.exists(preprocessor.checkpoint_dir):
        print("❌ No checkpoint directory found")
        return
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Copy all checkpoint files
    checkpoint_files = os.listdir(preprocessor.checkpoint_dir)
    
    if checkpoint_files:
        for file in checkpoint_files:
            src = os.path.join(preprocessor.checkpoint_dir, file)
            dst = os.path.join(backup_dir, file)
            shutil.copy2(src, dst)
        
        print(f"✅ Backed up {len(checkpoint_files)} checkpoint files to {backup_dir}")
    else:
        print("❌ No checkpoint files to backup")

def restore_checkpoints(preprocessor, backup_dir):
    """Restore checkpoint files from backup"""
    import shutil
    
    if not os.path.exists(backup_dir):
        print(f"❌ Backup directory not found: {backup_dir}")
        return
    
    backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.json')]
    
    if not backup_files:
        print("❌ No checkpoint files found in backup directory")
        return
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(preprocessor.checkpoint_dir, exist_ok=True)
    
    # Copy backup files
    for file in backup_files:
        src = os.path.join(backup_dir, file)
        dst = os.path.join(preprocessor.checkpoint_dir, file)
        shutil.copy2(src, dst)
    
    print(f"✅ Restored {len(backup_files)} checkpoint files from {backup_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Checkpoint Manager for Visual Backstory Preprocessing")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory containing checkpoints")
    parser.add_argument("--input_dir", default="data/visualcomet",
                        help="Input directory (needed for initialization)")
    parser.add_argument("--vcr_images_dir", default="",
                        help="VCR images directory (needed for initialization)")
    
    # Action arguments
    parser.add_argument("--status", action="store_true",
                        help="Show detailed checkpoint status")
    parser.add_argument("--list", action="store_true",
                        help="List checkpoint files")
    parser.add_argument("--clean", action="store_true",
                        help="Clean all checkpoints")
    parser.add_argument("--backup", type=str,
                        help="Backup checkpoints to specified directory")
    parser.add_argument("--restore", type=str,
                        help="Restore checkpoints from specified directory")
    
    args = parser.parse_args()
    
    # Initialize preprocessor (minimal initialization for checkpoint management)
    preprocessor = BackstoryPreprocessor(
        args.input_dir,
        args.output_dir,
        args.vcr_images_dir or "dummy",
        use_cuda=False  # Not needed for checkpoint management
    )
    
    # Execute requested action
    if args.status:
        show_detailed_status(preprocessor)
    elif args.list:
        list_checkpoint_files(preprocessor)
    elif args.clean:
        confirm = input("⚠️  Are you sure you want to clean all checkpoints? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            preprocessor.clean_checkpoints()
        else:
            print("❌ Operation cancelled")
    elif args.backup:
        backup_checkpoints(preprocessor, args.backup)
    elif args.restore:
        restore_checkpoints(preprocessor, args.restore)
    else:
        print("Please specify an action: --status, --list, --clean, --backup, or --restore")
        parser.print_help()

if __name__ == "__main__":
    main()
