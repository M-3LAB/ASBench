import os
import shutil
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import pandas as pd

# Input folder paths (only DRAEM and MemSeg)
folder_names = [
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/DRAEM_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/MemSeg_best_1"
]

# Output directory
output_root = "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/filer/filtered_draem_gt_memseg_diff_0.1"
os.makedirs(output_root, exist_ok=True)

def is_all_zeros_gt(gt_path):
    """Check if GT image is all zeros"""
    try:
        gt = np.array(Image.open(gt_path).convert('L'))
        return np.all(gt == 0)
    except:
        return True

def calculate_ap(anomaly_map_path, gt_path):
    """Calculate AP score"""
    if is_all_zeros_gt(gt_path):
        return np.nan
    try:
        anomaly_map = np.array(Image.open(anomaly_map_path).convert('L')).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32) / 255.0
        return average_precision_score(gt.flatten(), anomaly_map.flatten())
    except Exception as e:
        print(f"AP calculation error {anomaly_map_path}: {str(e)}")
        return np.nan

def process_image_pairs():
    """Main processing function"""
    stats = {
        'total_pairs': 0,
        'skipped_zero_gt': 0,
        'skipped_incomplete': 0,
        'skipped_draem_le_memseg': 0,
        'skipped_diff_le_0.1': 0,
        'matched_pairs': 0
    }
    
    results = []
    image_groups = {}
    
    # Collect all image data
    for folder in folder_names:
        model_name = os.path.basename(folder)
        for subcat in os.listdir(folder):
            subdir = os.path.join(folder, subcat)
            if not os.path.isdir(subdir):
                continue
            for img_file in os.listdir(subdir):
                if img_file.endswith('_anomaly_map.png'):
                    prefix = img_file.replace('_anomaly_map.png', '')
                    group_key = (subcat, prefix)
                    if group_key not in image_groups:
                        image_groups[group_key] = {}
                    gt_path = os.path.join(subdir, f'{prefix}_gt.png')
                    if is_all_zeros_gt(gt_path):
                        stats['skipped_zero_gt'] += 1
                        continue
                    img_data = {
                        'anomaly_map': os.path.join(subdir, f'{prefix}_anomaly_map.png'),
                        'gt': gt_path,
                        'heatmap_on_img': os.path.join(subdir, f'{prefix}_heatmap_on_img.png'),
                        'heatmap': os.path.join(subdir, f'{prefix}_heatmap_map.png'),
                        'origin': os.path.join(subdir, f'{prefix}_origin_img.png'),
                        'ap_score': None
                    }
                    image_groups[group_key][model_name] = img_data
    
    # Process each image pair
    stats['total_pairs'] = len(image_groups)
    for (subcat, prefix), model_dict in image_groups.items():
        # Check if both models are present
        if not all(m in model_dict for m in ["DRAEM_best_1", "MemSeg_best_1"]):
            stats['skipped_incomplete'] += 1
            continue
        
        # Calculate AP scores
        draem_ap = calculate_ap(
            model_dict["DRAEM_best_1"]['anomaly_map'],
            model_dict["DRAEM_best_1"]['gt']
        )
        memseg_ap = calculate_ap(
            model_dict["MemSeg_best_1"]['anomaly_map'],
            model_dict["MemSeg_best_1"]['gt']
        )
        
        # Check conditions
        if np.isnan(draem_ap) or np.isnan(memseg_ap):
            continue
            
        # Condition 1: DRAEM > MemSeg
        if draem_ap <= memseg_ap:
            stats['skipped_draem_le_memseg'] += 1
            continue
            
        # Condition 2: Difference > 0.1
        ap_diff = draem_ap - memseg_ap
        if ap_diff <= 0.1:
            stats['skipped_diff_le_0.1'] += 1
            continue
            
        # Save matching pairs
        stats['matched_pairs'] += 1
        output_dir = os.path.join(output_root, f"{subcat}_{prefix}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all images with appropriate naming
        for model, img_data in model_dict.items():
            role = "draem_" if model == "DRAEM_best_1" else "memseg_"
            for img_type in ['anomaly_map', 'gt', 'heatmap_on_img', 'heatmap', 'origin']:
                src = img_data[img_type]
                dst = os.path.join(output_dir, f"{role}{img_type}.png")
                shutil.copy2(src, dst)
        
        # Record results
        results.append({
            'Category': subcat,
            'Image': prefix,
            'MemSeg_AP': memseg_ap,
            'DRAEM_AP': draem_ap,
            'AP_Difference': ap_diff,
            'Output_Path': output_dir
        })
    
    # Save results
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('AP_Difference', ascending=False)
    df.to_csv(os.path.join(output_root, 'filtered_results.csv'), index=False)
    
    # Print statistics
    print(f"\n=== Processing Results ===")
    print(f"Total image pairs: {stats['total_pairs']}")
    print(f"Skipped (zero GT): {stats['skipped_zero_gt']}")
    print(f"Skipped (missing model): {stats['skipped_incomplete']}")
    print(f"Skipped (DRAEM ≤ MemSeg): {stats['skipped_draem_le_memseg']}")
    print(f"Skipped (difference ≤ 0.1): {stats['skipped_diff_le_0.1']}")
    print(f"Matched pairs: {stats['matched_pairs']}")
    print(f"\nResults saved to: {output_root}")
    
    # Print successful examples
    if not df.empty:
        print("\n=== Matched Pairs (sorted by AP difference) ===")
        for _, row in df.head(3).iterrows():
            print(f"\nCategory: {row['Category']} | Image: {row['Image']}")
            print(f"MemSeg AP: {row['MemSeg_AP']:.4f} < DRAEM AP: {row['DRAEM_AP']:.4f} (difference: +{row['AP_Difference']:.4f})")

if __name__ == "__main__":
    print("=== Start filtering image pairs ===")
    print("Filtering conditions:")
    print("1. DRAEM_best_1 > MemSeg_best_1")
    print("2. AP difference > 0.1")
    process_image_pairs()