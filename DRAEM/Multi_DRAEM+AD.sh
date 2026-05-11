#!/bin/bash

declare -a classes=("capsule" "bottle" "carpet" )
# "leather" "pill" "transistor" "tile" "cable" "zipper" "toothbrush" "metal_nut" "hazelnut"
# 从第六个类（索引为5）开始循环
for i in "${!classes[@]}"; do
  if [ $i -ge 0 ]; then 
    class=${classes[$i]}
    obj_id=$i
    echo "Training for class: $class with obj_id: $obj_id"
    
    CUDA_VISIBLE_DEVICES=3 python train_DRAEM.py \
      --obj_id $obj_id \
      --lr 0.0001 \
      --bs 8 \
      --epochs 500 \
      --data_path /cluster/home/zqyeleven/ASBenchmark/BTAD/ \
      --checkpoint_path checkpoints_Multi/Fractal+DRAEM+AD_BTAD \
      --log_path logs_Multi/Fractal+DRAEM+AD_BTAD \
      --Dataset_name Fractal+DRAEM+AD \
      --aug_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/$class/ko/image \
      --mask_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/$class/ko/mask \
      --origin_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/$class/ko/ori \
      --percent 0.5 > Multi_Result/Fractal+DRAEM+AD_BTAD/$class.log
  fi
done

wait
echo "All training tasks have been initiated."
# #!/bin/bash

# declare -a classes=("capsule" "bottle" "carpet" "leather" "pill" "tile" "cable" "toothbrush" "metal_nut" "hazelnut" "screw" "grid")

# # 从第六个类（索引为5）开始循环
# for i in "${!classes[@]}"; do
#   if [ $i -ge 0 ]; then  # 确保从第六个类开始
#     class=${classes[$i]}
#     obj_id=$i
#     echo "Training for class: $class with obj_id: $obj_id"
    
#     CUDA_VISIBLE_DEVICES=3 python train_DRAEM.py \
#       --obj_id $obj_id \
#       --lr 0.0001 \
#       --bs 8 \
#       --epochs 500 \
#       --data_path /cluster/home/zqyeleven/ASBenchmark/VisA_MVTec/ \
#       --checkpoint_path checkpoints_Multi/Fractal+DRAEM+AD_VisA \
#       --log_path logs_Multi/Fractal+DRAEM+AD_VisA \
#       --Dataset_name Fractal+DRAEM+AD \
#       --aug_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_VisA_data/$class/Anomaly/image \
#       --mask_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_VisA_data/$class/Anomaly/mask \
#       --origin_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/AD_VisA_data/$class/Anomaly/ori \
#       --percent 0.5 > Multi_Result/Fractal+DRAEM+AD_VisA/$class.log
#   fi
# done

# wait
# echo "All training tasks have been initiated."
# #!/bin/bash

# declare -a classes=("capsule" "bottle" "carpet" )
# # "leather" "pill" "transistor" "tile" "cable" "zipper" "toothbrush" "metal_nut" "hazelnut"
# # 从第六个类（索引为5）开始循环
# for i in "${!classes[@]}"; do
#   if [ $i -ge 0 ]; then 
#     class=${classes[$i]}
#     obj_id=$i
#     echo "Training for class: $class with obj_id: $obj_id"
    
#     CUDA_VISIBLE_DEVICES=3 python train_DRAEM.py \
#       --obj_id $obj_id \
#       --lr 0.0001 \
#       --bs 8 \
#       --epochs 500 \
#       --data_path /cluster/home/zqyeleven/ASBenchmark/BTAD/ \
#       --checkpoint_path checkpoints_Multi/Fractal+DRAEM+AD_BTAD \
#       --log_path logs_Multi/Fractal+DRAEM+AD_BTAD \
#       --Dataset_name Fractal+DRAEM+AD \
#       --aug_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/$class/ko/image \
#       --mask_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/$class/ko/mask \
#       --origin_dir /cluster/home/zqyeleven/ASBenchmark/destseg_perlin/generated_dataset/generated_dataset_BTAD/$class/ko/ori \
#       --percent 0.5 > Multi_Result/Fractal+DRAEM+AD_BTAD/$class.log
#   fi
# done

# wait
# echo "All training tasks have been initiated."