import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
import argparse


def load_trajectory_files(trajectory_files):
    """Load trajectory data from pickle files."""
    trajectories = {}
    for attack_name, filepath in trajectory_files.items():
        if os.path.exists(filepath):
            with open(filepath, "rb") as fp:
                trajectories[attack_name] = pickle.load(fp)
            print(f"Loaded {len(trajectories[attack_name])} trajectories from {attack_name}")
        else:
            print(f"Warning: {filepath} not found for attack {attack_name}")
    return trajectories


def get_image_sequence(trajectory_data):
    """Extract image sequence from trajectory data in order."""
    # Order: original, initialization, iteration_0, iteration_1, iteration_2, iteration_3, iteration_4, final
    images_dict = {stage: img for stage, img in trajectory_data['images']}
    
    sequence = []
    sequence.append(images_dict.get('original'))
    
    # Add initialization if available
    if 'initialization' in images_dict:
        sequence.append(images_dict['initialization'])
    
    # Add iterations 0-4 in order
    for i in range(5):
        key = f'iteration_{i}'
        if key in images_dict:
            sequence.append(images_dict[key])
        else:
            # If iteration is missing, use the last available iteration or final
            if sequence:
                sequence.append(sequence[-1])
            else:
                sequence.append(None)
    
    # Add final if available
    if 'final' in images_dict:
        sequence.append(images_dict['final'])
    elif sequence:
        # Use last iteration as final if final is missing
        sequence.append(sequence[-1])
    
    return sequence


def create_visualization(trajectories, output_file="attack_trajectories.png", max_images=5):
    """Create grid visualization of attack trajectories."""
    
    # Get the attack names
    attack_names = list(trajectories.keys())
    if not attack_names:
        print("ERROR: No trajectory data found!")
        return
    
    # Get number of images per attack
    num_images = min(max_images, min(len(trajectories[attack]) for attack in attack_names))
    
    # Each row shows one image's trajectory across attacks
    # Each column shows one stage (original, init, iter0-4, final)
    num_rows = num_images
    num_cols = 7  # original, initialization, iter0, iter1, iter2, iter3, iter4 (or final)
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2.5))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, 
                          hspace=0.15, wspace=0.1,
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # For each image
    for img_idx in range(num_images):
        # Get trajectory sequences for all attacks for this image
        sequences = {}
        for attack_name in attack_names:
            if img_idx < len(trajectories[attack_name]):
                trajectory_data = trajectories[attack_name][img_idx]
                sequences[attack_name] = get_image_sequence(trajectory_data)
        
        # Determine which attack to show (use first one, or combine)
        # For simplicity, show first attack's trajectory
        primary_attack = attack_names[0]
        if primary_attack not in sequences or not sequences[primary_attack]:
            continue
        
        sequence = sequences[primary_attack]
        
        # Plot each stage in the sequence
        for stage_idx in range(num_cols):
            ax = fig.add_subplot(gs[img_idx, stage_idx])
            ax.axis('off')
            
            if stage_idx < len(sequence) and sequence[stage_idx] is not None:
                img = sequence[stage_idx]
                if isinstance(img, Image.Image):
                    ax.imshow(np.array(img))
                else:
                    ax.imshow(img)
            
            # Add labels on first row
            if img_idx == 0:
                labels = ['Original', 'Init', 'Iter 0', 'Iter 1', 'Iter 2', 'Iter 3', 'Iter 4']
                if stage_idx < len(labels):
                    ax.set_title(labels[stage_idx], fontsize=10, pad=5)
        
        # Add image name on left side of first column
        if img_idx < len(trajectories[primary_attack]):
            trajectory_data = trajectories[primary_attack][img_idx]
            if 'image_name' in trajectory_data:
                fig.text(0.01, 1 - (img_idx + 0.5) / num_rows, 
                        trajectory_data['image_name'], 
                        ha='left', va='center', fontsize=9, rotation=0)
    
    # Add main title
    fig.suptitle(f'Attack Trajectories - {primary_attack}', fontsize=14, y=0.98)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Also save as PDF
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Visualization saved to: {pdf_file}")
    
    plt.close()


def create_multi_attack_visualization(trajectories, output_file="attack_trajectories_comparison.png", max_images=5):
    """Create visualization comparing multiple attacks side by side."""
    
    attack_names = list(trajectories.keys())
    if not attack_names:
        print("ERROR: No trajectory data found!")
        return
    
    num_images = min(max_images, min(len(trajectories[attack]) for attack in attack_names))
    num_attacks = len(attack_names)
    num_stages = 7  # original, init, iter0-4
    
    # Create figure: rows = images, columns = stages * attacks
    fig = plt.figure(figsize=(num_stages * num_attacks * 1.5, num_images * 2.5))
    gs = gridspec.GridSpec(num_images, num_stages * num_attacks, figure=fig,
                          hspace=0.15, wspace=0.05,
                          left=0.02, right=0.98, top=0.96, bottom=0.04)
    
    for img_idx in range(num_images):
        for attack_idx, attack_name in enumerate(attack_names):
            if img_idx >= len(trajectories[attack_name]):
                continue
            
            trajectory_data = trajectories[attack_name][img_idx]
            sequence = get_image_sequence(trajectory_data)
            
            # Plot each stage
            for stage_idx in range(num_stages):
                col_idx = attack_idx * num_stages + stage_idx
                ax = fig.add_subplot(gs[img_idx, col_idx])
                ax.axis('off')
                
                if stage_idx < len(sequence) and sequence[stage_idx] is not None:
                    img = sequence[stage_idx]
                    if isinstance(img, Image.Image):
                        ax.imshow(np.array(img))
                    else:
                        ax.imshow(img)
                
                # Add attack name and stage label on first row
                if img_idx == 0:
                    if stage_idx == 0:  # First column of each attack
                        ax.set_title(f'{attack_name}\nOriginal', fontsize=10, pad=5)
                    elif stage_idx == 1:
                        ax.set_title('Init', fontsize=9, pad=5)
                    elif stage_idx < 7:
                        ax.set_title(f'Iter {stage_idx-2}', fontsize=9, pad=5)
        
        # Add image name on left
        if img_idx < len(trajectories[attack_names[0]]):
            first_trajectory = trajectories[attack_names[0]][img_idx]
            if 'image_name' in first_trajectory:
                fig.text(0.01, 1 - (img_idx + 0.5) / num_images,
                        first_trajectory['image_name'],
                        ha='left', va='center', fontsize=9)
    
    fig.suptitle('Attack Trajectories Comparison', fontsize=16, y=0.99)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-attack visualization saved to: {output_file}")
    
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Multi-attack visualization saved to: {pdf_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize attack trajectories')
    parser.add_argument('--attack', type=str, help='Attack name (e.g., HSJA, CGBA)')
    parser.add_argument('--total-cost', type=float, default=10000.0, help='Total cost used in trajectory file')
    parser.add_argument('--query-cost', type=float, default=1.0, help='Query cost used in trajectory file')
    parser.add_argument('--search-cost', type=float, default=None, help='Search cost used in trajectory file')
    parser.add_argument('--overshooting', action='store_true', help='Overshooting flag used in trajectory file')
    parser.add_argument('--overshooting-scheduler-init', type=float, default=0.02, help='Overshooting scheduler init used')
    parser.add_argument('--output', type=str, default='attack_trajectories.png', help='Output file path')
    parser.add_argument('--max-images', type=int, default=5, help='Maximum number of images to visualize')
    parser.add_argument('--compare', action='store_true', help='Compare multiple attacks side by side')
    parser.add_argument('--attacks', type=str, nargs='+', help='List of attack names to compare (e.g., HSJA CGBA SURFREE)')
    
    args = parser.parse_args()
    
    # If comparing multiple attacks
    if args.compare and args.attacks:
        trajectory_files = {}
        for attack_name in args.attacks:
            filename = f"trajectories_{attack_name}_{args.total_cost}_{args.query_cost}_{args.search_cost}_{args.overshooting}_{args.overshooting_scheduler_init}.pkl"
            trajectory_files[attack_name] = filename
        
        trajectories = load_trajectory_files(trajectory_files)
        if trajectories:
            create_multi_attack_visualization(trajectories, args.output, args.max_images)
    elif args.attack:
        # Single attack visualization
        filename = f"trajectories_{args.attack}_{args.total_cost}_{args.query_cost}_{args.search_cost}_{args.overshooting}_{args.overshooting_scheduler_init}.pkl"
        trajectory_files = {args.attack: filename}
        trajectories = load_trajectory_files(trajectory_files)
        if trajectories:
            create_visualization(trajectories, args.output, args.max_images)
    else:
        # Try to find any trajectory files
        print("No attack specified. Looking for trajectory files...")
        trajectory_files = {}
        for attack_name in ['HSJA', 'CGBA', 'SURFREE', 'GEODA', 'OPT']:
            filename = f"trajectories_{attack_name}_{args.total_cost}_{args.query_cost}_{args.search_cost}_{args.overshooting}_{args.overshooting_scheduler_init}.pkl"
            if os.path.exists(filename):
                trajectory_files[attack_name] = filename
        
        if trajectory_files:
            if len(trajectory_files) > 1:
                trajectories = load_trajectory_files(trajectory_files)
                create_multi_attack_visualization(trajectories, args.output, args.max_images)
            else:
                attack_name = list(trajectory_files.keys())[0]
                trajectories = load_trajectory_files({attack_name: trajectory_files[attack_name]})
                create_visualization(trajectories, args.output, args.max_images)
        else:
            print("ERROR: No trajectory files found!")
            print("Usage: python visualize_trajectories.py --attack <ATTACK_NAME> [OPTIONS]")
            print("Or: python visualize_trajectories.py --compare --attacks HSJA CGBA SURFREE [OPTIONS]")


if __name__ == "__main__":
    main()

