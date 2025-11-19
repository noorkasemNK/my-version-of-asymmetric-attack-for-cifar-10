# Using Custom Images for Attack Trajectories

This guide shows you how to use your own images (like the 5 images you provided) with the attack visualization code.

## Quick Setup

### Step 1: Prepare Your Images

1. Create a folder called `custom_images` in the repository root:
   ```bash
   mkdir custom_images
   ```

2. Place your 5 images in that folder with these names:
   - `redbull.jpg` (or .jpeg, .png) - Red Bull can
   - `lemon.jpg` - Lemon
   - `banana_split.jpg` - Banana split
   - `soccer_ball.jpg` - Soccer ball
   - `orange_pen.jpg` - Orange pen

   **Note:** The images can have different extensions (.jpg, .jpeg, .png, .JPG, etc.) - the script will find them.

### Step 2: Run the Setup Script

```bash
python3 setup_custom_images.py
```

This script will:
- Find your images in the `custom_images` folder
- Convert and resize them to appropriate formats
- Copy them to the `data` directory with ImageNet-style naming
- Create a `val.txt` file with the proper format
- Back up your existing `val.txt` if it exists

### Step 3: Run Attacks with Trajectories

```bash
python3 main.py --attack HSJA --save-trajectories --device cpu --total-cost 10000 --image-start 1 --image-end 6
```

This will run attacks on your 5 custom images and save trajectory data.

### Step 4: Visualize the Trajectories

```bash
python3 visualize_trajectories.py --attack HSJA --total-cost 10000 --query-cost 1.0
```

This will create visualizations showing how each of your images is perturbed through the attack iterations.

## Alternative: Using Any Images

If you want to use different image names or more images:

1. Place any images in the `custom_images` folder
2. The script will automatically process all supported image files it finds
3. Run the same commands as above

## Image Requirements

- **Format**: JPG, JPEG, or PNG
- **Size**: Any size (will be automatically resized if too large)
- **Mode**: Will be converted to RGB automatically

## Troubleshooting

If the setup script doesn't find your images:
- Check that images are in the `custom_images` folder
- Verify file extensions are supported (.jpg, .jpeg, .png)
- Try using different file names that match the expected pattern

If you get errors during attacks:
- Make sure images were successfully copied to the `data` directory
- Check that `val.txt` was created correctly
- Verify images can be opened (not corrupted)


