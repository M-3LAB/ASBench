import wandb
import logging
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import ConcatDataset
from omegaconf import OmegaConf
from timm import create_model
from data import create_dataset, create_dataloader,create_dataset_RealNet
from data.dataset import MVTecDataset_AD,MVTecDataset,MVTecDataset_CutPaste,MVTecDataset_CutPaste_Scar,MVTecDataset_NSA,MVTecDataset_FPI,MVTecDataset_Fractal,MVTecDataset_CutOut,MVTecDataset_DFMGAN
from models import MemSeg, MemoryBank
from focal_loss import FocalLoss
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts
from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD
from train import evaluate,evaluate_2
_logger = logging.getLogger('train')


def run(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg.SEED)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.DATASET.target}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)

    
    # wandb
    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.EXP_NAME, project='MemSeg', config=OmegaConf.to_container(cfg))

    # build datasets
    if cfg.DATASETNAME == "MemSeg":
        print("MemSeg")
        trainset = create_dataset(
            datadir                = cfg.DATASET.datadir,
            target                 = cfg.DATASET.target, 
            is_train               = True,
            resize                 = cfg.DATASET.resize,
            imagesize              = cfg.DATASET.imagesize,
            texture_source_dir     = cfg.DATASET.texture_source_dir,
            structure_grid_size    = cfg.DATASET.structure_grid_size,
            transparency_range     = cfg.DATASET.transparency_range,
            perlin_scale           = cfg.DATASET.perlin_scale,
            min_perlin_scale       = cfg.DATASET.min_perlin_scale,
            perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
            use_mask               = cfg.DATASET.use_mask,
            bg_threshold           = cfg.DATASET.bg_threshold,
            bg_reverse             = cfg.DATASET.bg_reverse,
            percent                = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "RealNet":
        print("RealNet")
        trainset = create_dataset_RealNet(
            datadir                = cfg.DATASET.datadir,
            target                 = cfg.DATASET.target, 
            is_train               = True,
            resize                 = cfg.DATASET.resize,
            imagesize              = cfg.DATASET.imagesize,
            texture_source_dir     = '/opt/ml/code/luohan/sdas_MTD',
            structure_grid_size    = cfg.DATASET.structure_grid_size,
            transparency_range     = cfg.DATASET.transparency_range,
            perlin_scale           = cfg.DATASET.perlin_scale,
            min_perlin_scale       = cfg.DATASET.min_perlin_scale,
            perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
            use_mask               = cfg.DATASET.use_mask,
            bg_threshold           = cfg.DATASET.bg_threshold,
            bg_reverse             = cfg.DATASET.bg_reverse,
            percent                = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "DestSeg":
        print("DestSeg")
        trainset = MVTecDataset(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent=cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "CutPaste":
        print("CutPaste")
        trainset = MVTecDataset_CutPaste(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "AD":
        print("AD")
        trainset = MVTecDataset_AD(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            aug_dir = cfg.DATASET.aug_dir,
            origin_dir = cfg.DATASET.origin_dir,
            mask_dir = cfg.DATASET.mask_dir,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "DFMGAN":
        print("DFMGAN")
        trainset = MVTecDataset_DFMGAN(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            aug_dir = cfg.DATASET.aug_dir,
            origin_dir = cfg.DATASET.origin_dir,
            mask_dir = cfg.DATASET.mask_dir,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "CutPaste_Scar":
        print("CutPaste_Scar")
        trainset = MVTecDataset_CutPaste_Scar(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "NSA":
        print("NSA")
        trainset = MVTecDataset_NSA(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "FPI":
        print("FPI")
        trainset = MVTecDataset_FPI(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "Fractal":
        print("Fractal")
        trainset = MVTecDataset_Fractal(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    elif cfg.DATASETNAME == "CutOut":
        print("CutOut")
        trainset = MVTecDataset_CutOut(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    dataset_Fractal = MVTecDataset_Fractal(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent=cfg.DATASET.percent,
        )
    dataset_DRAEM = MVTecDataset(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent=cfg.DATASET.percent,
        )
    dataset_AD = MVTecDataset_AD(
            is_train=True,
            mvtec_dir=cfg.DATASET.datadir +"/"+ cfg.DATASET.target + "/train/good/",
            category=cfg.DATASET.target,
            aug_dir = cfg.DATASET.aug_dir,
            origin_dir = cfg.DATASET.origin_dir,
            mask_dir = cfg.DATASET.mask_dir,
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            dtd_dir=cfg.DATASET.texture_source_dir,
            rotate_90=False,
            random_rotate=0,
            percent = cfg.DATASET.percent,
        )
    dataset_MemSeg = create_dataset(
            datadir                = cfg.DATASET.datadir,
            target                 = cfg.DATASET.target, 
            is_train               = True,
            resize                 = cfg.DATASET.resize,
            imagesize              = cfg.DATASET.imagesize,
            texture_source_dir     = cfg.DATASET.texture_source_dir,
            structure_grid_size    = cfg.DATASET.structure_grid_size,
            transparency_range     = cfg.DATASET.transparency_range,
            perlin_scale           = cfg.DATASET.perlin_scale,
            min_perlin_scale       = cfg.DATASET.min_perlin_scale,
            perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
            use_mask               = cfg.DATASET.use_mask,
            bg_threshold           = cfg.DATASET.bg_threshold,
            bg_reverse             = cfg.DATASET.bg_reverse,
            percent                = cfg.DATASET.percent,
        )
    if(cfg.DATASETNAME == 'Fractal+DRAEM'):
        trainset=ConcatDataset([dataset_Fractal,dataset_DRAEM])
    elif(cfg.DATASETNAME == 'Fractal+AD'):
        print(cfg.DATASETNAME)
        trainset=ConcatDataset([dataset_Fractal,dataset_AD])
    elif(cfg.DATASETNAME == 'DRAEM+AD'):
        print(cfg.DATASETNAME)
        trainset=ConcatDataset([dataset_DRAEM,dataset_AD])
    elif(cfg.DATASETNAME == 'DRAEM+MemSeg'):
        print(cfg.DATASETNAME)
        trainset=ConcatDataset([dataset_DRAEM,dataset_MemSeg])
    elif(cfg.DATASETNAME == 'MemSeg+AD'):
        print(cfg.DATASETNAME)
        trainset=ConcatDataset([dataset_MemSeg,dataset_AD])
    elif(cfg.DATASETNAME == 'Fractal+MemSeg'):
        print(cfg.DATASETNAME)
        trainset=ConcatDataset([dataset_Fractal,dataset_MemSeg])
    elif(cfg.DATASETNAME == 'Fractal+MemSeg+DRAEM+AD'):
        print(cfg.DATASETNAME)
        trainset=ConcatDataset([dataset_Fractal,dataset_MemSeg,dataset_DRAEM,dataset_AD])
    elif(cfg.DATASETNAME == 'Fractal+MemSeg+DRAEM'):
        print(cfg.DATASETNAME)
        trainset = ConcatDataset([dataset_Fractal, dataset_MemSeg, dataset_DRAEM])
    elif(cfg.DATASETNAME == 'Fractal+MemSeg+AD'):
        print(cfg.DATASETNAME)
        trainset = ConcatDataset([dataset_Fractal, dataset_MemSeg, dataset_AD])
    elif(cfg.DATASETNAME == 'Fractal+DRAEM+AD'):
        print(cfg.DATASETNAME)
        trainset = ConcatDataset([dataset_Fractal, dataset_DRAEM, dataset_AD])
    elif(cfg.DATASETNAME == 'MemSeg+DRAEM+AD'):
        print(cfg.DATASETNAME)
        trainset = ConcatDataset([dataset_MemSeg, dataset_DRAEM, dataset_AD])



    print(cfg.DATASET.datadir + cfg.DATASET.target + "/train/good/")
    

    memoryset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = True,
        to_memory = True,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )

    testset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = False,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )
    
    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )


    # build feature extractor
    feature_extractor = create_model(
        cfg.MODEL.feature_extractor_name, 
        pretrained    = True, 
        features_only = True
    ).to(device)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    # build memory bank
    memory_bank = MemoryBank(
        normal_dataset   = memoryset,
        nb_memory_sample = cfg.MEMORYBANK.nb_memory_sample,
        device           = device
    )
    ## update normal samples and save
    memory_bank.update(feature_extractor=feature_extractor)
    torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    _logger.info('Update {} normal samples in memory bank'.format(cfg.MEMORYBANK.nb_memory_sample))

    # build MemSeg
    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor
    ).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma = cfg.TRAIN.focal_gamma, 
        alpha = cfg.TRAIN.focal_alpha
    )

    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, model.parameters()), 
        lr           = cfg.OPTIMIZER.lr, 
        weight_decay = cfg.OPTIMIZER.weight_decay
    )

    if cfg['SCHEDULER']['use_scheduler']:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            first_cycle_steps = cfg.TRAIN.num_training_steps,
            max_lr = cfg.OPTIMIZER.lr,
            min_lr = cfg.SCHEDULER.min_lr,
            warmup_steps   = int(cfg.TRAIN.num_training_steps * cfg.SCHEDULER.warmup_ratio)
        )
    else:
        scheduler = None

    # Fitting model
    training(
        model              = model, 
        num_training_steps = cfg.TRAIN.num_training_steps, 
        trainloader        = trainloader, 
        validloader        = testloader, 
        criterion          = [l1_criterion, f_criterion], 
        loss_weights       = [cfg.TRAIN.l1_weight, cfg.TRAIN.focal_weight],
        optimizer          = optimizer,
        scheduler          = scheduler,
        log_interval       = cfg.LOG.log_interval,
        eval_interval      = cfg.LOG.eval_interval,
        savedir            = savedir,
        device             = device,
        use_wandb          = cfg.TRAIN.use_wandb,
        target             = cfg.DATASET.target
    )
    
def eval(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg.SEED)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.DATASET.target}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)

    
    # wandb
    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.EXP_NAME, project='MemSeg', config=OmegaConf.to_container(cfg))

    # build datasets

    memoryset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = True,
        to_memory = True,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )

    testset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = False,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )
    
    # build dataloader
    
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        # batch_size  = cfg.DATALOADER.batch_size,
        batch_size  = 1,
        num_workers = cfg.DATALOADER.num_workers
    )

    # build feature extractor
    feature_extractor = create_model(
        cfg.MODEL.feature_extractor_name, 
        pretrained    = True, 
        features_only = True
    ).to(device)
    ## freeze weight of layer1,2,3
    for l in ['layer1','layer2','layer3']:
        for p in feature_extractor[l].parameters():
            p.requires_grad = False

    # build memory bank
    memory_bank = MemoryBank(
        normal_dataset   = memoryset,
        nb_memory_sample = cfg.MEMORYBANK.nb_memory_sample,
        device           = device
    )
    ## update normal samples and save
    memory_bank.update(feature_extractor=feature_extractor)
    torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    _logger.info('Update {} normal samples in memory bank'.format(cfg.MEMORYBANK.nb_memory_sample))

    # build MemSeg
    model = MemSeg(
        memory_bank       = memory_bank,
        feature_extractor = feature_extractor
    ).to(device)
    
    # torch.load(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
    model.load_state_dict(torch.load(os.path.join(savedir, f'best_model.pt')))
    model.eval()

    # Fitting model
    eval_metrics = evaluate_2(
        path         = cfg.DATASET.anomalymap_path,
        model        = model, 
        dataloader   = testloader, 
        device       = device,
        sub_dataset  = cfg.DATASET.target,
    )
    eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
    print(eval_log)


if __name__=='__main__':
    args = OmegaConf.from_cli()
    # load default config
    cfg = OmegaConf.load(args.configs)
    del args['configs']
    
    # merge config with new keys
    cfg = OmegaConf.merge(cfg, args)
    
    # target cfg
    target_cfg = OmegaConf.load(cfg.DATASET.anomaly_mask_info)
    cfg.DATASET = OmegaConf.merge(cfg.DATASET, target_cfg[cfg.DATASET.target])
    
    print(OmegaConf.to_yaml(cfg))

    # run(cfg)
    eval(cfg)
