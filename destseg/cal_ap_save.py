import os
import shutil
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import pandas as pd

# 输入文件夹路径（保持不变）
folder_names = [
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/AD_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/DRAEM_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/MemSeg_best_1",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+DRAEM+AD_mvtec",
    "/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/vis_multi/Fractal+MemSeg+AD_mvtec"
]

# 需要验证的关键模型分组
base_models = ["MemSeg_best_1", "Fractal_best_1", "DRAEM_best_1", "AD_best_1"]
combo_models = ["Fractal+DRAEM+AD_mvtec", "Fractal+MemSeg+AD_mvtec"]
all_required_models = base_models + combo_models

# 输出目录
output_root = "/cluster/home/zqyeleven/ASBenchmark/grouped_validation"
os.makedirs(output_root, exist_ok=True)

def is_all_zeros_gt(gt_path):
    """检查GT图像是否全为0"""
    try:
        gt = np.array(Image.open(gt_path).convert('L'))
        return np.all(gt == 0)
    except:
        return True  # 如果无法读取视为无效

def calculate_ap(anomaly_map_path, gt_path):
    """计算AP分数（自动跳过全零GT）"""
    if is_all_zeros_gt(gt_path):
        return np.nan
    try:
        anomaly_map = np.array(Image.open(anomaly_map_path).convert('L')).astype(np.float32) / 255.0
        gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32) / 255.0
        return average_precision_score(gt.flatten(), anomaly_map.flatten())
    except Exception as e:
        print(f"AP计算错误 {anomaly_map_path}: {str(e)}")
        return np.nan

def validate_scores(base_scores, combo_scores):
    """验证所有base_score < 所有combo_score且差值至少0.05"""
    if any(np.isnan(s) for s in base_scores + combo_scores):
        return False
    max_base = max(base_scores)
    min_combo = min(combo_scores)
    return (max_base < min_combo) and ((min_combo - max_base) >= 0.05)

def process_image_groups():
    """主处理函数"""
    stats = {
        'total_groups': 0,
        'skipped_zero_gt': 0,
        'skipped_incomplete': 0,
        'skipped_condition_fail': 0,
        'matched_groups': 0
    }
    
    results = []
    image_groups = {}

    # 阶段1：收集所有图片数据
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
                        'ap_score': None  # 稍后计算
                    }
                    image_groups[group_key][model_name] = img_data

    # 阶段2：处理每个图片组
    stats['total_groups'] = len(image_groups)
    for (subcat, prefix), model_dict in image_groups.items():
        # 检查是否包含所有必要模型
        present_models = set(model_dict.keys())
        if not all(m in present_models for m in all_required_models):
            stats['skipped_incomplete'] += 1
            continue

        # 计算AP分数
        base_scores = []
        combo_scores = []
        for model in base_models:
            if model_dict[model]['ap_score'] is None:
                model_dict[model]['ap_score'] = calculate_ap(
                    model_dict[model]['anomaly_map'],
                    model_dict[model]['gt']
                )
            base_scores.append(model_dict[model]['ap_score'])
        
        for model in combo_models:
            if model_dict[model]['ap_score'] is None:
                model_dict[model]['ap_score'] = calculate_ap(
                    model_dict[model]['anomaly_map'],
                    model_dict[model]['gt']
                )
            combo_scores.append(model_dict[model]['ap_score'])

        # 验证条件
        if validate_scores(base_scores, combo_scores):
            stats['matched_groups'] += 1
            difference = min(combo_scores) - max(base_scores)
            output_dir = os.path.join(output_root, f"{subcat}_{prefix}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存所有图片（基础模型标记为base_前缀）
            for model, img_data in model_dict.items():
                prefix_tag = "base_" if model in base_models else "combo_" if model in combo_models else ""
                for img_type in ['anomaly_map', 'gt', 'heatmap_on_img', 'heatmap', 'origin']:
                    src = img_data[img_type]
                    dst = os.path.join(output_dir, f"{prefix_tag}{model}_{img_type}.png")
                    shutil.copy2(src, dst)

            # 记录结果
            results.append({
                'Category': subcat,
                'Image': prefix,
                'Max_Base_AP': max(base_scores),
                'Min_Combo_AP': min(combo_scores),
                'Difference': difference,
                'Base_Scores': " | ".join(f"{m}:{s:.4f}" for m,s in zip(base_models, base_scores)),
                'Combo_Scores': " | ".join(f"{m}:{s:.4f}" for m,s in zip(combo_models, combo_scores)),
                'Output_Path': output_dir
            })
        else:
            stats['skipped_condition_fail'] += 1

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_root, 'validation_results.csv'), index=False)

    # 打印统计信息
    print(f"\n=== 处理结果统计 ===")
    print(f"总图片组数: {stats['total_groups']}")
    print(f"跳过全零GT组: {stats['skipped_zero_gt']}")
    print(f"跳过不完整组（缺少关键模型）: {stats['skipped_incomplete']}")
    print(f"跳过条件不满足组: {stats['skipped_condition_fail']}")
    print(f"成功匹配组 (差值≥0.05): {stats['matched_groups']}")
    print(f"\n结果已保存至: {output_root}")

    # 打印成功示例
    if not df.empty:
        print("\n=== 匹配成功的组示例 (差值≥0.05) ===")
        for _, row in df.head(3).iterrows():
            print(f"\n类别: {row['Category']} | 图像: {row['Image']}")
            print(f"基础模型最高AP: {row['Max_Base_AP']:.4f} < 组合模型最低AP: {row['Min_Combo_AP']:.4f}")
            print(f"差异值: {row['Difference']:.4f}")
            print("基础模型分数:")
            print(row['Base_Scores'])
            print("组合模型分数:")
            print(row['Combo_Scores'])

if __name__ == "__main__":
    print("=== 开始验证图片组 ===")
    print("验证条件要求:")
    print("1. 以下基础模型的AP分数全部小于组合模型:")
    print("   - 基础模型:", ", ".join(base_models))
    print("   - 组合模型:", ", ".join(combo_models))
    print("2. 基础模型之间、组合模型之间无需顺序")
    print("3. 组合模型最小值必须比基础模型最大值至少高0.05")
    process_image_groups()