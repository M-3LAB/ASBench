# CUDA_VISIBLE_DEVICES=0 python train.py --gpu_id 0 --num_workers 16  --custom_training_category --no_rotation_category bottle --checkpoint_path saved_model_Multi/Fractal_AD_mpdd --mvtec_path /cluster/home/zqyeleven/ASBenchmark/mpdd_MVTec/ --Dataset DRAEM+AD --percent 1 --aug_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_mpdd_data/bottle/Anomaly/image --mask_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_mpdd_data/bottle/Anomaly/mask --origin_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_mpdd_data/bottle/Anomaly/ori > Multi_Result/DRAEM+AD_mpdd/bottle

# #!/bin/bash

# # 定义要处理的类别列表
# categories=("bottle" "capsule" "carpet" "cable" "grid" )

# # 定义通用参数
# GPU_ID=0
# NUM_WORKERS=16
# CHECKPOINT_PATH="saved_model_Multi/DRAEM+AD_mpdd"
# MVTEC_PATH="/cluster/home/zqyeleven/ASBenchmark/mpdd/"
# DATASET="DRAEM+AD"
# PERCENT=1
# AUG_DIR_BASE="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/MPDD_AD_data/AUG/"
# MASK_DIR_BASE="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/MPDD_AD_data/AUG/"
# ORIGIN_DIR_BASE="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/MPDD_AD_data/AUG/"
# RESULT_DIR_BASE="Multi_Result/DRAEM+AD_mpdd"

# # 创建结果目录（如果不存在）
# mkdir -p $RESULT_DIR_BASE

# # 遍历每个类别并执行训练命令
# for category in "${categories[@]}"
# do
#     echo "开始训练类别: $category"

#     # 定义各类别的目录
#     AUG_DIR="$AUG_DIR_BASE/$category/image"
#     MASK_DIR="$MASK_DIR_BASE/$category/mask"
#     ORIGIN_DIR="$ORIGIN_DIR_BASE/$category/ori"
#     LOG_FILE="$RESULT_DIR_BASE/$category.log"

#     # 运行训练命令
#     CUDA_VISIBLE_DEVICES=2 python train.py \
#         --gpu_id $GPU_ID \
#         --num_workers $NUM_WORKERS \
#         --custom_training_category \
#         --no_rotation_category $category \
#         --checkpoint_path $CHECKPOINT_PATH \
#         --mvtec_path $MVTEC_PATH \
#         --Dataset $DATASET \
#         --percent $PERCENT \
#         --aug_dir $AUG_DIR \
#         --mask_dir $MASK_DIR \
#         --origin_dir $ORIGIN_DIR \
#         > $LOG_FILE 2>&1 

#     # 可选：等待部分训练完成，避免过多并行任务占用资源
#     # sleep 10

#     echo "训练命令已提交，日志将记录在 $LOG_FILE"
# done

# # 等待所有后台任务完成
# wait

# echo "所有训练任务已完成。"
#!/bin/bash

# 定义要处理的类别列表
categories=("bottle" "cable" "capsule" "carpet" "grid" "hazelnut" "leather" "metal_nut" "pill" "screw" "tile" "toothbrush")

# 定义通用参数
GPU_ID=0
NUM_WORKERS=16
CHECKPOINT_PATH="saved_model_Multi/DRAEM+AD_VisA"
MVTEC_PATH="/cluster/home/zqyeleven/ASBenchmark/VisA_MVTec/"
DATASET="DRAEM+AD"
PERCENT=1
AUG_DIR_BASE="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_VisA_data"
MASK_DIR_BASE="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_VisA_data"
ORIGIN_DIR_BASE="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_VisA_data"
RESULT_DIR_BASE="Multi_Result/DRAEM+AD_VisA"

# 创建结果目录（如果不存在）
mkdir -p $RESULT_DIR_BASE

# 遍历每个类别并执行训练命令
for category in "${categories[@]}"
do
    echo "开始训练类别: $category"

    # 定义各类别的目录
    AUG_DIR="$AUG_DIR_BASE/$category/Anomaly/image"
    MASK_DIR="$MASK_DIR_BASE/$category/Anomaly/mask"
    ORIGIN_DIR="$ORIGIN_DIR_BASE/$category/Anomaly/ori"
    LOG_FILE="$RESULT_DIR_BASE/$category.log"

    # 运行训练命令
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --gpu_id $GPU_ID \
        --num_workers $NUM_WORKERS \
        --custom_training_category \
        --no_rotation_category $category \
        --checkpoint_path $CHECKPOINT_PATH \
        --mvtec_path $MVTEC_PATH \
        --Dataset $DATASET \
        --percent $PERCENT \
        --aug_dir $AUG_DIR \
        --mask_dir $MASK_DIR \
        --origin_dir $ORIGIN_DIR \
        --visualization_path visualization_path/DRAEM+AD_VisA/$category \
        > $LOG_FILE 

    # 可选：等待部分训练完成，避免过多并行任务占用资源
    # sleep 10

    echo "训练命令已提交，日志将记录在 $LOG_FILE"
done

# 等待所有后台任务完成
wait

echo "所有训练任务已完成。"
