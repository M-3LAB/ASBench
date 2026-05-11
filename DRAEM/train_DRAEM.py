import torch
from data_loader import MVTecDRAEMTrainDataset, MVTecDataset_DestSeg, MVTecDataset_MemSeg, MVTecDRAEMTestDataset,MVTecDataset_CutOut,MVTecDataset_Fractal,MVTecDataset_FPI,  MVTecDataset_NSA, MVTecDataset_CutPaste,MVTecDataset_CutPaste_Scar,MVTecDataset_RealNet,MVTec_Anomaly_Detection,MVTecDRAEMTrainDataset_AD,MVTecDRAEMTrainDataset_DFMGAN
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import ConcatDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluation(obj_name, model, model_seg, dataset, dataloader):
    with torch.no_grad():
        model.eval()
        model_seg.eval()
        img_dim = 256
        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        for i_batch, sample_batched in enumerate(dataloader):

            gray_batch = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            gray_rec= model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        print(obj_name + "AUC Image:  " +str(auroc) +" , AP Image:  " +str(ap) + " , AUC Pixel:  " +str(auroc_pixel) + " , AP Pixel:  " +str(ap_pixel))
    return auroc,ap,auroc_pixel,ap_pixel


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(obj_names, args, rotate_90 = False, random_rotate = 0):
    print(args.data_path)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    for obj_name in obj_names:
        best_value=0
        print(obj_name)
        print(args.Dataset_name)
        print(args.percent)
        run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

        visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)
        model.train()
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)
        model_seg.train()
        
        optimizer = torch.optim.Adam([
                                      {"params": model.parameters(), "lr": args.lr},
                                      {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()
        
        testDataset = MVTecDRAEMTestDataset(args.data_path + obj_name + "/test/", resize_shape=[256, 256])
        testDataloader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=0) 
        
        if(args.Dataset_name == 'DRAEM'):
            dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path,args.percent, resize_shape=[256, 256])
        elif(args.Dataset_name == 'AD'):
            dataset = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir, resize_shape=[256, 256])
        elif(args.Dataset_name == 'DFMGAN'):
            dataset = MVTecDRAEMTrainDataset_DFMGAN(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir_DFM,args.origin_dir_DFM,args.mask_dir_DFM, resize_shape=[256, 256])
        elif(args.Dataset_name == 'DestSeg'):
            dataset = MVTecDataset_DestSeg(
                is_train=True,
                mvtec_dir=args.data_path + obj_name + "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
                percent=args.percent,
            )
        elif(args.Dataset_name == 'CutOut'):
            dataset = MVTecDataset_CutOut(
                is_train=True,
                mvtec_dir=args.data_path + obj_name+ "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
                percent = args.percent,
            )
        elif(args.Dataset_name=='Fractal'):
            dataset = MVTecDataset_Fractal(
                is_train=True,
                mvtec_dir=args.data_path + obj_name+ "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                percent = args.percent,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
            )
        elif(args.Dataset_name=='FPI'):
            dataset = MVTecDataset_FPI(
                is_train=True,
                mvtec_dir=args.data_path + obj_name+ "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                percent = args.percent,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
            )
        elif(args.Dataset_name=='NSA'):
            dataset = MVTecDataset_NSA(
                is_train=True,
                mvtec_dir=args.data_path + obj_name+ "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                percent = args.percent,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
                category=obj_name,
            )
        elif(args.Dataset_name=='CutPaste'):
            dataset = MVTecDataset_CutPaste(
                is_train=True,
                mvtec_dir=args.data_path + obj_name+ "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
                percent = args.percent,
            )
        elif(args.Dataset_name=='MemSeg'):
            dataset = MVTecDataset_MemSeg(
                datadir                = args.data_path,
                target                 = obj_name,
                is_train               = True,
                to_memory              = False,
                resize                 = [256, 256],
                imagesize              = 256,
                texture_source_dir     = args.anomaly_source_path, 
                structure_grid_size    = 8,
                transparency_range     = [0, 0.8],
                perlin_scale           = 6, 
                min_perlin_scale       = 0, 
                perlin_noise_threshold = 0.5,
                use_mask               = False,
                bg_threshold           = None,
                bg_reverse             = None,
                percent = args.percent,
            )
        elif(args.Dataset_name=='RealNet'):
            dataset = MVTecDataset_RealNet(
                datadir                = args.data_path,
                target                 = obj_name,
                is_train               = True,
                to_memory              = False,
                resize                 = [256, 256],
                imagesize              = 256,
                texture_source_dir     = '/opt/ml/code/luohan/sdas_MTD', 
                structure_grid_size    = 8,
                transparency_range     = [0.5, 1.0],
                perlin_scale           = 6, 
                min_perlin_scale       = 0, 
                perlin_noise_threshold = 0.5,
                use_mask               = False,
                bg_threshold           = None,
                bg_reverse             = None,
                percent = args.percent,
            )
        elif(args.Dataset_name=='CutPaste_Scar'):
            dataset = MVTecDataset_CutPaste_Scar(
                is_train=True,
                mvtec_dir=args.data_path + obj_name+ "/train/good/",
                resize_shape=RESIZE_SHAPE,
                normalize_mean=NORMALIZE_MEAN,
                normalize_std=NORMALIZE_STD,
                dtd_dir=args.anomaly_source_path,
                rotate_90=rotate_90,
                random_rotate=random_rotate,
                percent = args.percent,
            )
        elif(args.Dataset_name=='AD'):
            dataset = MVTec_Anomaly_Detection(args,obj_name,length=500)
            
            
        dataset_fractal = MVTecDataset_Fractal(
            is_train=True,
            mvtec_dir=args.data_path + obj_name+ "/train/good/",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=args.anomaly_source_path,
            rotate_90=rotate_90,
            random_rotate=random_rotate,
            percent = args.percent,
        )
        dataset_DRAEM = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path,args.percent, resize_shape=[256, 256])
        dataset_MemSeg = MVTecDataset_MemSeg(
                datadir                = args.data_path,
                target                 = obj_name,
                is_train               = True,
                to_memory              = False,
                resize                 = [256, 256],
                imagesize              = 256,
                texture_source_dir     = args.anomaly_source_path, 
                structure_grid_size    = 8,
                transparency_range     = [0, 0.8],
                perlin_scale           = 6, 
                min_perlin_scale       = 0, 
                perlin_noise_threshold = 0.5,
                use_mask               = False,
                bg_threshold           = None,
                bg_reverse             = None,
                percent = args.percent,
            )
#         
#         dataset_DFMGAN = MVTecDRAEMTrainDataset_DFMGAN(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir_DFM,args.origin_dir_DFM,args.mask_dir_DFM, resize_shape=[256, 256])
        if(args.Dataset_name == 'Fractal+DRAEM'):
            dataset=ConcatDataset([dataset_fractal,dataset_DRAEM])
        elif(args.Dataset_name == 'DRAEM+MemSeg'):
            dataset=ConcatDataset([dataset_MemSeg,dataset_DRAEM])
        elif(args.Dataset_name == 'DRAEM+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_DRAEM,dataset_AD])
        elif(args.Dataset_name == 'Fractal+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_fractal,dataset_AD])
        elif(args.Dataset_name == 'MemSeg+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_MemSeg,dataset_AD])
        elif(args.Dataset_name == 'Fractal+MemSeg'):
            print(args.Dataset_name)
            dataset=ConcatDataset([dataset_fractal,dataset_MemSeg])
        elif(args.Dataset_name == 'Fractal+DRAEM+MemSeg+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_fractal,dataset_DRAEM,dataset_MemSeg,dataset_AD])
        elif(args.Dataset_name == 'Fractal+DRAEM+MemSeg'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_fractal,dataset_DRAEM,dataset_MemSeg])
        elif(args.Dataset_name == 'Fractal+DRAEM+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_fractal,dataset_DRAEM,dataset_AD])
        elif(args.Dataset_name == 'Fractal+MemSeg+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_fractal,dataset_MemSeg,dataset_AD])
        elif(args.Dataset_name == 'DRAEM+MemSeg+AD'):
            print(args.Dataset_name)
            dataset_AD = MVTecDRAEMTrainDataset_AD(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, args.percent,args.aug_dir,args.origin_dir,args.mask_dir,resize_shape=[256, 256])
            dataset=ConcatDataset([dataset_DRAEM,dataset_MemSeg,dataset_AD])
            
        dataloader = DataLoader(dataset, batch_size=args.bs,
                                shuffle=True, num_workers=16)
        generator = torch.Generator().manual_seed((0))

        n_iter = 0
        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                gray_rec = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                # print(out_mask_sm.shape,anomaly_mask.shape)
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = l2_loss + ssim_loss + segment_loss

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                if args.visualize and n_iter % 400 == 0:
                    t_mask = out_mask_sm[:, 1:, :, :]
                    visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                    visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                    visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                    visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')


                n_iter +=1

            scheduler.step()

#             torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
#             torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))
            if epoch % 10 == 0 and epoch != 0:
                model.eval()
                model_seg.eval()
                auroc,ap,auroc_pixel,ap_pixel=evaluation(obj_name, model, model_seg, testDataset, testDataloader)
                if auroc+ap_pixel>=best_value :
                    best_value = auroc+ap_pixel
                    print(best_value)
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
                    torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))
                model.train()
                model_seg.train()


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int)
    parser.add_argument('--lr', action='store', type=float)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, default='/opt/ml/code/luohan/VisA_MVTec/')
    parser.add_argument('--anomaly_source_path', action='store', type=str, default='datasets/dtd/images/')
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--Dataset_name',action='store',type=str,required=True)
    parser.add_argument('--percent',type=float,default=0.5,required=True)
    parser.add_argument('--aug_dir',type=str,default='/opt/ml/code/luohan/VisA_MVTec')
    parser.add_argument('--origin_dir',type=str,default='/opt/ml/code/luohan/VisA_MVTec')
    parser.add_argument('--mask_dir',type=str,default='/opt/ml/code/luohan/VisA_MVTec')
    parser.add_argument('--aug_dir_DFM',type=str,default='/opt/ml/code/luohan/VisA_MVTec')
    parser.add_argument('--origin_dir_DFM',type=str,default='/opt/ml/code/luohan/VisA_MVTec')
    parser.add_argument('--mask_dir_DFM',type=str,default='/opt/ml/code/luohan/VisA_MVTec')
    args = parser.parse_args()
    setup_seed(444)
    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['tile'],
                 ['cable'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid']
                 ]

    if int(args.obj_id) == -1:
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -2:
        obj_list = ['capsule',
                     'bottle',
                     'carpet',
                     'leather',
                     'pill',
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -3:
        obj_list = [ 'metal_nut',
                      'hazelnut',
                      'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -100:
        obj_list = [ 'capsule',
                     'bottle',
                     'carpet',
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -6:
        obj_list = ['metal_nut',
                     'hazelnut',
                     'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -7:
        obj_list = [ 
                    'capsule',
                     'bottle',
                    'carpet',
                     'grid',
                     'cable',
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -8:
        obj_list = [
                     'transistor',
                     'tile',
                     'cable',
                     'zipper',
                     'toothbrush',
                     'metal_nut',
                     'hazelnut',
                     ]
        picked_classes = obj_list
    elif int(args.obj_id) == -9:
        obj_list = [ 'screw',
                     'grid',
                     'wood'
                     ]
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)


# import torch
# from data_loader import MVTecDRAEMTrainDataset,MVTecDataset_CutPaste_Scar,MVTecDataset_MemSeg,MVTecDataset_DestSeg,MVTecDataset_CutOut,MVTecDataset_Fractal,MVTecDataset_FPI,MVTecDataset_NSA,MVTecDataset_CutPaste
# from torch.utils.data import DataLoader
# from torch import optim
# from tensorboard_visualizer import TensorboardVisualizer
# from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
# from loss import FocalLoss, SSIM
# import os
# from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

# def train_on_device(obj_names, args,rotate_90=False, random_rotate=0):

#     if not os.path.exists(args.checkpoint_path):
#         os.makedirs(args.checkpoint_path)

#     if not os.path.exists(args.log_path):
#         os.makedirs(args.log_path)

#     for obj_name in obj_names:
#         print(obj_name)
#         print(args.Dataset_name)
#         run_name = 'DRAEM_test_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+obj_name+'_'

#         visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

#         model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
#         model.cuda()
#         model.apply(weights_init)

#         model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
#         model_seg.cuda()
#         model_seg.apply(weights_init)

#         optimizer = torch.optim.Adam([
#                                       {"params": model.parameters(), "lr": args.lr},
#                                       {"params": model_seg.parameters(), "lr": args.lr}])

#         scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)

#         loss_l2 = torch.nn.modules.loss.MSELoss()
#         loss_ssim = SSIM()
#         loss_focal = FocalLoss()
#         if(args.Dataset_name=='DRAEM'):
#             dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/good/", args.anomaly_source_path, resize_shape=[256, 256])
#         elif(args.Dataset_name=='DestSeg'):
#             dataset = MVTecDataset_DestSeg(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         elif(args.Dataset_name=='CutOut'):
#             dataset = MVTecDataset_CutOut(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         elif(args.Dataset_name=='Fractal'):
#             dataset = MVTecDataset_Fractal(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         elif(args.Dataset_name=='FPI'):
#             dataset = MVTecDataset_FPI(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         elif(args.Dataset_name=='NSA'):
#             dataset = MVTecDataset_NSA(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         elif(args.Dataset_name=='CutPaste'):
#             dataset = MVTecDataset_CutPaste(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         elif(args.Dataset_name=='MemSeg'):
#             dataset = MVTecDataset_MemSeg(
#                 datadir                = args.data_path,
#                 target                 = obj_name,
#                 is_train               = True,
#                 to_memory              = False,
#                 resize                 = [256, 256],
#                 imagesize              = 256,
#                 texture_source_dir     = args.anomaly_source_path, 
#                 structure_grid_size    = 8,
#                 transparency_range     = [0.15, 1.],
#                 perlin_scale           = 6, 
#                 min_perlin_scale       = 0, 
#                 perlin_noise_threshold = 0.5,
#                 use_mask               = True,
#                 bg_threshold           = 100,
#                 bg_reverse             = False
#             )
#         elif(args.Dataset_name=='CutPaste_Scar'):
#             dataset = MVTecDataset_CutPaste_Scar(
#                 is_train=True,
#                 mvtec_dir=args.data_path + obj_name+ "/train/good/",
#                 resize_shape=RESIZE_SHAPE,
#                 normalize_mean=NORMALIZE_MEAN,
#                 normalize_std=NORMALIZE_STD,
#                 dtd_dir=args.anomaly_source_path,
#                 rotate_90=rotate_90,
#                 random_rotate=random_rotate,
#             )
#         dataloader = DataLoader(dataset, batch_size=args.bs,
#                                 shuffle=True, num_workers=16)

#         n_iter = 0
#         for epoch in range(args.epochs):
#             print("Epoch: "+str(epoch))
#             for i_batch, sample_batched in enumerate(dataloader):
#                 gray_batch = sample_batched["image"].cuda()
#                 aug_gray_batch = sample_batched["augmented_image"].cuda()
#                 anomaly_mask = sample_batched["anomaly_mask"].cuda()
#                 gray_rec = model(aug_gray_batch)
#                 joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

#                 out_mask = model_seg(joined_in)
#                 out_mask_sm = torch.softmax(out_mask, dim=1)
                
#                 l2_loss = loss_l2(gray_rec,gray_batch)
#                 ssim_loss = loss_ssim(gray_rec, gray_batch)
#                 # print(out_mask_sm.shape,anomaly_mask.shape)
#                 segment_loss = loss_focal(out_mask_sm, anomaly_mask)
#                 loss = l2_loss + ssim_loss + segment_loss

#                 optimizer.zero_grad()

#                 loss.backward()
#                 optimizer.step()

#                 if args.visualize and n_iter % 200 == 0:
#                     visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
#                     visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
#                     visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
#                 if args.visualize and n_iter % 400 == 0:
#                     t_mask = out_mask_sm[:, 1:, :, :]
#                     visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
#                     visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
#                     visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
#                     visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
#                     visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')


#                 n_iter +=1

#             scheduler.step()

#             torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
#             torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))


# if __name__=="__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--obj_id', action='store', type=int, required=True)
#     parser.add_argument('--bs', action='store', type=int, required=True)
#     parser.add_argument('--lr', action='store', type=float, required=True)
#     parser.add_argument('--epochs', action='store', type=int, required=True)
#     parser.add_argument('--gpu_id', action='store', type=int, default=2, required=False)
#     parser.add_argument('--data_path', action='store', type=str, default='/opt/ml/code/luohan/ml-destseg-main_REB/datasets/mvtec/')
#     parser.add_argument('--anomaly_source_path', action='store', type=str, default='/opt/ml/code/luohan/ml-destseg-main_REB/datasets/dtd/images/')
#     parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
#     parser.add_argument('--log_path', action='store', type=str, required=True)
#     parser.add_argument('--visualize', action='store_true')
#     parser.add_argument('--Dataset_name',action='store',type=str,required=True)

#     args = parser.parse_args()

#     obj_batch = [['capsule'],
#                  ['bottle'],
#                  ['carpet'],
#                  ['leather'],
#                  ['pill'],
#                  ['transistor'],
#                  ['tile'],
#                  ['cable'],
#                  ['zipper'],
#                  ['toothbrush'],
#                  ['metal_nut'],
#                  ['hazelnut'],
#                  ['screw'],
#                  ['grid'],
#                  ['wood']
#                  ]

#     if int(args.obj_id) == -1:
#         obj_list = ['capsule',
#                      'bottle',
#                      'carpet',
#                      'leather',
#                      'pill',
#                      'transistor',
#                      'tile',
#                      'cable',
#                      'zipper',
#                      'toothbrush',
#                      'metal_nut',
#                      'hazelnut',
#                      'screw',
#                      'grid',
#                      'wood'
#                      ]
#         picked_classes = obj_list
#     elif int(args.obj_id) == -2:
#         obj_list = [ #'transistor',
#                      'tile',
#                      'cable',
#                      'zipper',
#                      'toothbrush'
#                      ]
#         picked_classes = obj_list
#     elif int(args.obj_id) == -3:
#         obj_list = [ 'metal_nut',
#                       'hazelnut',
#                       'screw',
#                      'grid',
#                      'wood'
#                      ]
#         picked_classes = obj_list
#     elif int(args.obj_id) == -4:
#         obj_list = [# 'capsule',
#                      'bottle',
#                      'carpet',
#                      'leather',
#                      'pill'
#                      ]
#         picked_classes = obj_list
#     elif int(args.obj_id) == -5:
#         obj_list = [ 'capsule',
#                      'bottle',
#                      'carpet',
#                      ]
#         picked_classes = obj_list
#     else:
#         picked_classes = obj_batch[int(args.obj_id)]

#     with torch.cuda.device(args.gpu_id):
#         train_on_device(picked_classes, args)

