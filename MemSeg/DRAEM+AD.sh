# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py configs=configs.yaml EXP_NAME=DRAEM+AD DATASET.target=bottle DATASETNAME=DRAEM+AD DATASET.percent=0.5 RESULT.savedir=./saved_model_multi/DRAEM+AD_BTAD  DATASET.datadir=/cluster/home/zqyeleven/ASBenchmark/BTAD DATASET.aug_dir=/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/capsule/ko/image DATASET.origin_dir=/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/capsule/ko/ori DATASET.mask_dir=/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/capsule/ko/mask > Multi_Result/DRAEM+AD_BTAD/bottle 


#!/bin/bash

# 定义类别列表
categories=("bottle" "cable" "capsule" "carpet" "grid"  )

# 定义基础路径
mvtec_path="/cluster/home/zqyeleven/ASBenchmark/mpdd/"
aug_base_dir="/cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/MPDD_AD_data/AUG/"
result_dir="Multi_Result/DRAEM+AD_MPDD"
savedir_base="./saved_model_multi/DRAEM+AD_MPDD"

# 创建结果目录（如果尚未创建）
mkdir -p $result_dir

# 遍历每个类别，生成并执行训练命令
for category in "${categories[@]}"
do
    echo "正在为类别 $category 运行训练命令..."
    CUDA_VISIBLE_DEVICES=2 nohup python -u main.py \
        configs=configs.yaml \
        EXP_NAME=DRAEM+AD \
        DATASET.target=$category \
        DATASETNAME=DRAEM+AD \
        DATASET.percent=0.5 \
        RESULT.savedir=${savedir_base}/${category} \
        DATASET.datadir=${mvtec_path} \
        DATASET.aug_dir=${aug_base_dir}/${category}/image \
        DATASET.origin_dir=${aug_base_dir}/${category}/ori \
        DATASET.mask_dir=${aug_base_dir}/${category}/mask \
        > ${result_dir}/${category}.log 2>&1 
done

echo "所有训练命令已启动。"
