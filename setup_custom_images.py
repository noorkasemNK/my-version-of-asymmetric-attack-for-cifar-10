#!/usr/bin/env python3
"""
Helper script to set up custom images for attack experiments.
This script helps you prepare your custom images for use with the attack code.
"""

import os
import shutil
from PIL import Image
import argparse


def setup_custom_images(image_dir="custom_images", data_dir="data"):
    """
    Set up custom images for attack experiments.
    
    Args:
        image_dir: Directory containing your custom images
        data_dir: Target data directory for the repository
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Expected image files (based on your 5 images) - optional reference
    expected_files = [
        "redbull.jpg",  # Red Bull can
        "lemon.jpg",    # Lemon
        "banana_split.jpg",  # Banana split
        "soccer_ball.jpg",   # Soccer ball
        "orange_pen.jpg"     # Orange pen
    ]
    
    # Check if images exist in the image_dir
    print(f"Looking for images in: {image_dir}/")
    
    # First, try to find expected files
    found_images = []
    
    # Check for expected files
    print("\nChecking for expected image files:")
    for img_file in expected_files:
        base_name = img_file.replace('.jpg', '')
        source_path = None
        found_name = None
        
        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.JPEG', '.png', '.PNG', '.JPG']:
            test_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(test_path):
                source_path = test_path
                found_name = os.path.basename(test_path)
                print(f"  ✓ Found: {found_name}")
                found_images.append((img_file, source_path, found_name))
                break
    
    # If no expected files found, look for any image files in the directory
    if not found_images:
        print("\nNo expected files found. Searching for any image files...")
        supported_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
        all_files = os.listdir(image_dir) if os.path.exists(image_dir) else []
        
        for filename in sorted(all_files):
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext in supported_extensions:
                    target_name = f"image_{len(found_images) + 1:03d}.jpg"
                    print(f"  ✓ Found: {filename}")
                    found_images.append((target_name, file_path, filename))
                    
                    # Limit to reasonable number
                    if len(found_images) >= 10:
                        print("  (Limited to first 10 images)")
                        break
    
    if not found_images:
        print(f"\n⚠️  No images found in {image_dir}/")
        print("\nPlease:")
        print(f"1. Create a directory called '{image_dir}'")
        print("2. Place your 5 images in it with these names:")
        for img_file in image_files:
            print(f"   - {img_file}")
        print("\nSupported formats: .jpg, .jpeg, .png")
        return False
    
    # Copy and rename images to ImageNet format
    print(f"\n📋 Setting up images in {data_dir}/...")
    
    # Create a custom val.txt file
    val_lines = []
    
    for idx, (target_name, source_path, original_name) in enumerate(found_images, start=1):
        # Rename to ImageNet format
        new_name = f"ILSVRC2012_val_{idx:08d}.JPEG"
        target_path = os.path.join(data_dir, new_name)
        
        try:
            # Open and convert image
            img = Image.open(source_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize if too large (optional, for faster processing)
            max_size = 224 * 2  # 448px
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            img.save(target_path, 'JPEG', quality=95)
            
            # Add to val.txt with a dummy label (we'll use label 1 as placeholder)
            # The actual label doesn't matter much for visualization
            val_lines.append(f"{new_name} 1\n")
            
            print(f"  ✓ Saved: {original_name} -> {target_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {source_path}: {e}")
            continue
    
    # Write val.txt
    val_txt_path = os.path.join(data_dir, "val.txt")
    
    # Backup existing val.txt if it exists
    if os.path.exists(val_txt_path):
        backup_path = val_txt_path + ".backup"
        shutil.copy2(val_txt_path, backup_path)
        print(f"\n  ℹ️  Backed up existing val.txt to {backup_path}")
    
    # Write new val.txt with just our custom images
    with open(val_txt_path, 'w') as f:
        f.writelines(val_lines)
    
    print(f"\n  ✓ Created {val_txt_path} with {len(val_lines)} entries")
    print(f"\n✅ Setup complete! {len(found_images)} images ready for attacks.")
    print(f"\nNow you can run:")
    print(f"  python3 main.py --attack HSJA --save-trajectories --device cpu --total-cost 10000 --image-start 1 --image-end {len(found_images) + 1}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Set up custom images for attack experiments')
    parser.add_argument('--image-dir', type=str, default='custom_images',
                       help='Directory containing your custom images (default: custom_images)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Target data directory (default: data)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Custom Images Setup for Asymmetric Attacks")
    print("=" * 60)
    
    setup_custom_images(args.image_dir, args.data_dir)


if __name__ == "__main__":
    main()

