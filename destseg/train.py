import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset,MVTecDataset_CutOut,MVTecDataset_NSA,MVTecDataset_FPI,MemSegDataset,MVTecDRAEMTrainDataset,MVTecDataset_CutPaste_Scar,MemSegDataset_RealNet,MVTecDataset_AD,MVTecDataset_CutPaste,MVTecDataset_Fractal
from eval import evaluate
from model.destseg import DeSTSeg
from model.losses import cosine_similarity_loss, focal_loss, l1_loss
from torch.utils.data import ConcatDataset

warnings.filterwarnings("ignore")


def train(args, category, rotate_90=False, random_rotate=0):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(os.path.join(args.checkpoint_path,"best")):
        os.makedirs(os.path.join(args.checkpoint_path,"best"))
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"{args.run_name_head}_{args.steps}_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = DeSTSeg(dest=True, ed=True).cuda()

    seg_optimizer = torch.optim.SGD(
        [
            {"params": model.segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": model.segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    de_st_optimizer = torch.optim.SGD(
        [
            {"params": model.student_net.parameters(), "lr": args.lr_de_st},
        ],
        lr=0.4,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )
    if args.Dataset_name =="DestSeg":
        dataset = MVTecDataset(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name =="CutOut":
        dataset = MVTecDataset_CutOut(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name =="CutPaste":
        dataset = MVTecDataset_CutPaste(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name =="Fractal":
        dataset = MVTecDataset_Fractal(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name =="AD":
        dataset = MVTecDataset_AD(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            ad_dir='AD_data/wood_sort',
            ad_mask_dir='AD_data/wood_mask_sort',
            ad_ori_dir='AD_data/wood_origin_sort',
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name =="CutPaste_Scar":
        dataset = MVTecDataset_CutPaste_Scar(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name =="NSA":
        dataset = MVTecDataset_NSA(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    elif args.Dataset_name=="Fractal":
        dataset = MVTecDataset_Fractal(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
        )
    elif args.Dataset_name =="FPI":
        dataset = MVTecDataset_FPI(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
        )
    elif args.Dataset_name == "MemSeg":
        print("Mem")
        dataset = MemSegDataset(
            datadir                = args.mvtec_path,
            target                 = category, 
            is_train                  = True,
            to_memory              = False,
            category               = category,
            resize                 = (256,256),
            imagesize              = 256,
            texture_source_dir     = args.dtd_path, 
            structure_grid_size    = 8,
            transparency_range     = [0.15, 1.],
            perlin_scale           = 6, 
            min_perlin_scale       = 0, 
            perlin_noise_threshold = 0.5,
            use_mask               = True,
            bg_threshold           = 100,
            bg_reverse             = True,
            percent                = args.percent
        )
    elif args.Dataset_name == "RealNet":
        dataset = MemSegDataset_RealNet(
            datadir                = args.mvtec_path,
            target                 = category, 
            is_train                  = True,
            to_memory              = False,
            category               = category,
            resize                 = (256,256),
            imagesize              = 256,
            texture_source_dir     = '/cluster/home/zqyeleven/RealNet-main/data/MVTec-AD/sdas/', 
            structure_grid_size    = 8,
            transparency_range     = [0.15, 1.],
            perlin_scale           = 6, 
            min_perlin_scale       = 0, 
            perlin_noise_threshold = 0.5,
            use_mask               = True,
            bg_threshold           = 100,
            bg_reverse             = True,
            percent                = args.percent
        )
    elif args.Dataset_name == "DRAEM":
        print("DRAEM")
        dataset = MVTecDRAEMTrainDataset(args.mvtec_path + category + "/train/good/", args.dtd_path,category = category,percent=args.percent, resize_shape=[256, 256])
    
    dataset_Fractal = MVTecDataset_Fractal(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.dtd_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    dataset_DRAEM = MVTecDRAEMTrainDataset(args.mvtec_path + category + "/train/good/", args.dtd_path,category = category,percent=args.percent, resize_shape=[256, 256])   
    dataset_AD = MVTecDataset_AD(
            is_train=True,
            mvtec_dir=args.mvtec_path + category + "/train/good/",
            category=category,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            ad_dir=args.aug_dir,
            ad_mask_dir=args.mask_dir,
            ad_ori_dir=args.origin_dir,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent=args.percent,
        )
    dataset_MemSeg = MemSegDataset(
            datadir                = args.mvtec_path,
            target                 = category, 
            is_train                  = True,
            to_memory              = False,
            category               = category,
            resize                 = (256,256),
            imagesize              = 256,
            texture_source_dir     = args.dtd_path, 
            structure_grid_size    = 8,
            transparency_range     = [0.15, 1.],
            perlin_scale           = 6, 
            min_perlin_scale       = 0, 
            perlin_noise_threshold = 0.5,
            use_mask               = True,
            bg_threshold           = 100,
            bg_reverse             = True,
            percent                = args.percent
        )
#         dataset_DFMGAN = MVTecDRAEMTrainDataset_DFMGAN(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir_DFM,args.origin_dir_DFM,args.mask_dir_DFM, resize_shape=[256, 256])
    if(args.Dataset_name == 'Fractal+DRAEM'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_DRAEM])
    elif(args.Dataset_name == 'Fractal+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_AD])
    elif(args.Dataset_name == 'DRAEM+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_DRAEM,dataset_AD])
    elif(args.Dataset_name == 'DRAEM+MemSeg'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_DRAEM,dataset_MemSeg])
    elif(args.Dataset_name == 'MemSeg+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_MemSeg,dataset_AD])
    elif(args.Dataset_name == 'Fractal+MemSeg'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_MemSeg])
    elif(args.Dataset_name == 'Fractal+MemSeg+DRAEM+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_MemSeg,dataset_DRAEM,dataset_AD])
    elif(args.Dataset_name == 'Fractal+DRAEM+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_DRAEM,dataset_AD])
    elif(args.Dataset_name == 'MemSeg+DRAEM+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_MemSeg,dataset_DRAEM,dataset_AD])
    elif(args.Dataset_name == 'Fractal+MemSeg+AD'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_MemSeg,dataset_AD])
    elif(args.Dataset_name == 'Fractal+MemSeg+DRAEM'):
        print(args.Dataset_name)
        dataset=ConcatDataset([dataset_Fractal,dataset_MemSeg,dataset_DRAEM])
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0

    flag = True
    
    best_value = 0
    while flag:
        for _, sample_batched in enumerate(dataloader):
            seg_optimizer.zero_grad()
            de_st_optimizer.zero_grad()
            img_origin = sample_batched["img_origin"].cuda()
            img_aug = sample_batched["img_aug"].cuda()
            mask = sample_batched["mask"].cuda()
            
            # print("img_origin",img_origin.shape)
            # print(img_aug.shape)
            # print(mask.shape)
            # print("type",type(img_origin),type(img_aug),type(mask))
            if global_step < args.de_st_steps:
                model.student_net.train()
                model.segmentation_net.eval()
            else:
                model.student_net.eval()
                model.segmentation_net.train()

            output_segmentation, output_de_st, output_de_st_list = model(
                img_aug, img_origin
            )

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            cosine_loss_val = cosine_similarity_loss(output_de_st_list)
            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)

            if global_step < args.de_st_steps:
                total_loss_val = cosine_loss_val
                total_loss_val.backward()
                de_st_optimizer.step()
            else:
                total_loss_val = focal_loss_val + l1_loss_val
                total_loss_val.backward()
                seg_optimizer.step()

            global_step += 1

            visualizer.add_scalar("cosine_loss", cosine_loss_val, global_step)
            visualizer.add_scalar("focal_loss", focal_loss_val, global_step)
            visualizer.add_scalar("l1_loss", l1_loss_val, global_step)
            visualizer.add_scalar("total_loss", total_loss_val, global_step)

            if global_step % args.eval_per_steps == 0:
                auc_detect_seg,ap_seg=evaluate(args, category, model, visualizer, global_step)
                if auc_detect_seg + ap_seg >=best_value:
                    best_value=auc_detect_seg + ap_seg
                    print("best_value",best_value)
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_path,"best", run_name + ".pckl"))

            if global_step % args.log_per_steps == 0:
                if global_step < args.de_st_steps:
                    print(
                        f"Training at global step {global_step}, cosine loss: {round(float(cosine_loss_val), 4)}"
                    )
                else:
                    print(
                        f"Training at global step {global_step}, focal loss: {round(float(focal_loss_val), 4)}, l1 loss: {round(float(l1_loss_val), 4)}"
                    )

            if global_step >= args.steps:
                flag = False
                break

    torch.save(
        model.state_dict(), os.path.join(args.checkpoint_path, run_name + ".pckl")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="/cluster/home/zqyeleven/destseg_perlin/datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="DeSTSeg_MVTec")
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--Dataset_name",type=str,required=True)
    parser.add_argument("--percent",type=float,required=True)
    
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000) #5000
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=100)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    parser.add_argument('--aug_dir',type=str,default='AD_data/wood_sort')
    parser.add_argument('--origin_dir',type=str,default='AD_data/wood_sort')
    parser.add_argument('--mask_dir',type=str,default='AD_data/wood_sort')
    parser.add_argument('--aug_dir_DFM',type=str,default='AD_data/wood_sort')
    parser.add_argument('--origin_dir_DFM',type=str,default='AD_data/wood_sort')
    parser.add_argument('--mask_dir_DFM',type=str,default='AD_data/wood_sort')
    # parser.add_argument("--visualization_path", nargs="*", type=str, default="visualization_path")
    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
             "toothbrush",
            "transistor",
        ]
        slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
        ]
        rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
        ]
    
    with torch.cuda.device(args.gpu_id):
        for obj in no_rotation_category:
            print(obj)
            train(args, obj)

        for obj in slight_rotation_category:
            print(obj)
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            train(args, obj, rotate_90=True, random_rotate=5)
