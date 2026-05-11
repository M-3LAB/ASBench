import os
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd

# Define folder paths (same as before)
folder_names = [
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/AD_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/DRAEM_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/MemSeg_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/DRAEM+MemSeg_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+DRAEM+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+MemSeg+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+MemSeg+DRAEM_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/MemSeg+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/DRAEM+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+DRAEM_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+MemSeg+DRAEM+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+MemSeg_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/MemSeg+DRAEM+AD_mvtec",
]

# Define the exact saving order as specified
model_saving_order = [
    "AD_best_1",
    "MemSeg_best_1",
    "Fractal_best_1",
    "DRAEM+MemSeg_mvtec",
    "Fractal+AD_mvtec",
    "DRAEM_best_1",
    "Fractal+MemSeg_mvtec",
    "Fractal+DRAEM_mvtec",
    "Fractal+MemSeg+DRAEM_mvtec",
    "MemSeg+DRAEM+AD_mvtec",
    "MemSeg+AD_mvtec",
    "Fractal+MemSeg+DRAEM+AD_mvtec",
    "DRAEM+AD_mvtec",
    "Fractal+DRAEM+AD_mvtec",
    "Fractal+MemSeg+AD_mvtec"
]

# Output directory
output_root = "/cluster/home/zqyeleven/ASBenchmark/ordered_results"  # Change this to your desired output path
os.makedirs(output_root, exist_ok=True)

def calculate_ap(anomaly_map_path, gt_path):
    """Calculate AP score for an image pair"""
    try:
        anomaly_map = np.array(Image.open(anomaly_map_path).convert('L')).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32) / 255.0
        return average_precision_score(gt.flatten(), anomaly_map.flatten())
    except Exception as e:
        print(f"Error calculating AP for {anomaly_map_path}: {str(e)}")
        return np.nan

def process_and_save_ordered_images():
    """Main function to process and save images in specified order"""
    all_images = {}
    results = []
    
    # First collect all image groups
    for folder in folder_names:
        model_name = os.path.basename(folder)
        
        for subcat in os.listdir(folder):
            subdir = os.path.join(folder, subcat)
            if not os.path.isdir(subdir):
                continue
                
            for img_file in os.listdir(subdir):
                if img_file.endswith('_anomaly_map.png'):
                    prefix = img_file.replace('_anomaly_map.png', '')
                    key = (subcat, prefix)
                    
                    if key not in all_images:
                        all_images[key] = []
                    
                    # Get all related image files
                    img_paths = {
                        'model': model_name,
                        'anomaly_map': os.path.join(subdir, f'{prefix}_anomaly_map.png'),
                        'gt': os.path.join(subdir, f'{prefix}_gt.png'),
                        'heatmap_on_img': os.path.join(subdir, f'{prefix}_heatmap_on_img.png'),
                        'heatmap': os.path.join(subdir, f'{prefix}_heatmap_map.png'),
                        'origin': os.path.join(subdir, f'{prefix}_origin_img.png')
                    }
                    
                    # Calculate AP score
                    img_paths['ap_score'] = calculate_ap(img_paths['anomaly_map'], img_paths['gt'])
                    all_images[key].append(img_paths)
    
    # Now save the images in specified order
    for (subcat, prefix), images in all_images.items():
        # Create output directory for this image group
        output_dir = os.path.join(output_root, subcat, prefix)
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort images according to the specified order
        sorted_images = sorted(images, key=lambda x: model_saving_order.index(x['model']))
        
        # Save each image in order
        for img_data in sorted_images:
            model = img_data['model']
            
            # Save all image types
            for img_type, src_path in img_data.items():
                if img_type in ['model', 'ap_score']:
                    continue
                    
                dest_filename = f"{model}_{img_type}_{os.path.basename(src_path)}"
                dest_path = os.path.join(output_dir, dest_filename)
                shutil.copy2(src_path, dest_path)
            
            # Record results
            results.append({
                'Subcategory': subcat,
                'Image': prefix,
                'Model': model,
                'AP_Score': img_data['ap_score'],
                'Output_Path': output_dir
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_root, 'ordered_results_summary.csv'), index=False)
    return df

if __name__ == "__main__":
    print("Starting ordered image saving process...")
    print(f"Models will be saved in this exact order: {model_saving_order}")
    
    results_df = process_and_save_ordered_images()
    
    print("\nProcess completed successfully!")
    print(f"Results saved to: {output_root}")
    print(f"Summary CSV saved to: {os.path.join(output_root, 'ordered_results_summary.csv')}")
    
    # Print verification of order
    sample_group = next(iter(results_df.groupby(['Subcategory', 'Image'])))
    print("\nSample group order verification:")
    print(sample_group[1][['Model', 'AP_Score']].to_string(index=False))