import src.models.utils as model_utils
from src.models.model import ModelWrapper
from utils.data import open_image, open_labels, get_label_by_name, check_image, get_mean_std
from utils.args import get_args, get_attack
from tqdm import tqdm
import pickle
import os

args = get_args()
labels = open_labels(args.data_dir)
mean, std = get_mean_std(model_name=args.model, device=args.device)
model = ModelWrapper(model_utils.get_model(model_name=args.model).to(device=args.device), mean, std)
attack = get_attack(args=args, model=model)
logs = []
trajectories = []

for i in tqdm(range(args.image_start, args.image_end), desc="Running The Attack"):
    image_name = f"{args.image_prefix}{'%08d' % i}.JPEG"
    image_path = f"{args.data_dir}/{image_name}"
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"\nWarning: Image file not found: {image_path}")
        print("Please make sure ImageNet validation images are in the data directory.")
        continue
    
    try:
        image_label = get_label_by_name(labels, image_name)
    except:
        print(f"\nWarning: Label not found for {image_name}, skipping...")
        continue
    
    try:
        image = model_utils.preprocess(open_image(source_image=image_path), mean=model.mean, std=model.std, model_name=args.model).to(args.device)
    except Exception as e:
        print(f"\nWarning: Could not load image {image_path}: {e}")
        continue
    
    # Check if model correctly classifies the image
    # If not, use the actual predicted label (for custom images that might not match val.txt labels)
    predicted_label = model(image)
    if predicted_label != image_label:
        print(f"\nNote: Image {image_name} predicted as {predicted_label} but label file says {image_label}")
        print(f"  Using predicted label {predicted_label} for attack")
        image_label = predicted_label
    
    perturbed_image, current_cost = attack(image, image_label)
    # print(attack.logs)
    logs.append([image_name, attack.logs])
    
    # Save trajectory images if enabled
    if args.save_trajectories:
        if hasattr(attack, 'trajectory_images') and attack.trajectory_images:
            trajectory_data = {
                'image_name': image_name,
                'images': []
            }
            try:
                for stage, img_tensor in attack.trajectory_images:
                    # Convert tensor to PIL Image
                    img_pil = model_utils.postprocess(img_tensor, mean=model.mean, std=model.std, squeeze=True)
                    trajectory_data['images'].append((stage, img_pil))
                trajectories.append(trajectory_data)
                print(f"  ✓ Saved trajectory for {image_name} ({len(trajectory_data['images'])} stages)")
            except Exception as e:
                print(f"  ✗ Error saving trajectory for {image_name}: {e}")
        else:
            print(f"  ⚠ Warning: No trajectory images found for {image_name}")

    # Save logs after each image (updated incrementally)
    if args.save_logs:
        with open(f"attack_{args.attack}_{args.total_cost}_{args.query_cost}_{args.search_cost}_{args.overshooting}_{args.overshooting_scheduler_init}.log", "wb") as fp:
            pickle.dump(logs, fp)

# Save trajectories after all images are processed
if args.save_trajectories and trajectories:
    trajectory_file = f"trajectories_{args.attack}_{args.total_cost}_{args.query_cost}_{args.search_cost}_{args.overshooting}_{args.overshooting_scheduler_init}.pkl"
    with open(trajectory_file, "wb") as fp:
        pickle.dump(trajectories, fp)
    print(f"\n✅ Trajectories saved to: {trajectory_file}")