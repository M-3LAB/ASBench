import time
import json
import os 
import wandb
import logging

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List
from sklearn.metrics import roc_auc_score,average_precision_score
from metrics import compute_pro, trapezoid
import cv2

_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def training(model, trainloader, validloader, criterion, optimizer, scheduler,target, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4], 
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, device: str ='cpu') -> dict:   

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    l1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights
    
    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True
    while train_mode:

        end = time.time()
        for inputs, masks, targets,image_path in trainloader:
            # batch
            inputs, masks, targets= inputs.to(device), masks.to(device), targets.to(device)
            
            data_time_m.update(time.time() - end)

            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            l1_loss = l1_criterion(outputs[:,1,:], masks)
            focal_loss = focal_criterion(outputs, masks)
            loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)

            loss.backward()
            
            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())
            
            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                wandb.log({
                    'lr':optimizer.param_groups[0]['lr'],
                    'train_focal_loss':focal_losses_m.val,
                    'train_l1_loss':l1_losses_m.val,
                    'train_loss':losses_m.val
                },
                step=step)
            
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'L1 Loss: {l1_loss.val:>6.4f} ({l1_loss.avg:>6.4f}) '
                            'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            step+1, num_training_steps, 
                            loss       = losses_m, 
                            l1_loss    = l1_losses_m,
                            focal_loss = focal_losses_m,
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))


            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps: 
                eval_metrics = evaluate(
                    model        = model, 
                    dataloader   = validloader, 
                    device       = device,
                    # sub_dataset  = target
                )
                model.train()

                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    wandb.log(eval_log, step=step)

                # checkpoint
                # if best_score < np.mean(list(eval_metrics.values())):
                if best_score < eval_metrics['AUROC-image']+eval_metrics['AP-pixel']:
                    # save best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, eval_metrics['AUROC-image']+eval_metrics['AP-pixel']))

                    best_score = eval_metrics['AUROC-image']+eval_metrics['AP-pixel']

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')


def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    image=cv2.resize(image,(256,256))
    heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def evaluate(model, dataloader, device: str = 'cpu'):
    # targets and outputs
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []

    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets,image_path) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

            # stack targets and outputs
            image_targets.extend(targets.cpu().tolist())
            image_masks.extend(masks.cpu().numpy())
            
            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(outputs[:,1,:].cpu().numpy())
            
    # metrics    
    image_masks = np.array(image_masks)
    anomaly_map = np.array(anomaly_map)
    auroc_image = roc_auc_score(image_targets, anomaly_score)
    ap_image = average_precision_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    ap_pixel =  average_precision_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    all_fprs, all_pros = compute_pro(
        anomaly_maps      = anomaly_map,
        ground_truth_maps = image_masks
    )
    aupro = trapezoid(all_fprs, all_pros)
    print(f"AUROC (Image Level): {auroc_image}")
    print(f"AP (Image Level): {ap_image}")
    print(f"AUROC (Pixel Level): {auroc_pixel}")
    print(f"AP (Pixel Level): {ap_pixel}")
    print(f"PRO (Pixel Level): {aupro}")
    metrics = {
        'AUROC-image':auroc_image,
        'AUROC-pixel':auroc_pixel,
        'AP-image':ap_image,
        'AP-pixel':ap_pixel,
        'AUPRO-pixel':aupro

    }
    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%% | AP-pixel: %.3f%% | AP-image: %.3f%%'  % 
                (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel'], metrics['AP-pixel'], metrics['AP-image']))
    return metrics
        
def evaluate_2(path,model, dataloader, device: str = 'cpu', sub_dataset = 'bootle'):
    # targets and outputs
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []

    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets,image_path) in enumerate(dataloader):
        # for idx, (inputs, masks, targets, img_path) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            #print(image_path[0])
            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

            # stack targets and outputs
            image_targets.extend(targets.cpu().tolist())
            image_masks.extend(masks.cpu().numpy())
            
            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(outputs[:,1,:].cpu().numpy())
            # if(idx==0):
            #     print(inputs.shape) # bchw 1chw-> hwc
            #     print(masks.shape) # bhw 1hw-> hw
            #     print(outputs.shape) # b2hw 12hw-> hw
            if(not os.path.exists(os.path.join(path,sub_dataset))):
                os.mkdir(os.path.join(path,sub_dataset))
            
            # input_img = cv.imread(img_path)
            anomaly_map_cur = outputs[0,1,:].cpu().numpy()*255
            # input_img = inputs.squeeze(0).permute(1,2,0).cpu().numpy()*255
            heatmap = cv2heatmap(anomaly_map_cur)
            
            cv2.imwrite(os.path.join(path,sub_dataset,f"{idx:03d}"+'_anomaly_map.png'),anomaly_map_cur)
            cv2.imwrite(os.path.join(path,sub_dataset,f"{idx:03d}"+'_heatmap.png'),heatmap)
            origin_image = cv2.imread(image_path[0])
            heatmap_on_img = heatmap_on_image(heatmap, origin_image)
            origin_image = cv2.resize(origin_image, (256, 256))
            cv2.imwrite(os.path.join(path,sub_dataset,f"{idx:03d}"+'_origin_image.png'),origin_image)
            cv2.imwrite(os.path.join(path,sub_dataset,f"{idx:03d}"+'_heatmap_on_img.png'),heatmap_on_img)
            cv2.imwrite(os.path.join(path,sub_dataset,f"{idx:03d}"+'_gt.png'),masks.squeeze(0).cpu().numpy()*255)
            
    # metrics    
    image_masks = np.array(image_masks)
    anomaly_map = np.array(anomaly_map)
    
    auroc_image = roc_auc_score(image_targets, anomaly_score)
    ap_image = average_precision_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    ap_pixel =  average_precision_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    all_fprs, all_pros = compute_pro(
        anomaly_maps      = anomaly_map,
        ground_truth_maps = image_masks
    )
    aupro = trapezoid(all_fprs, all_pros)
    print(f"AUROC (Image Level): {auroc_image}")
    print(f"AP (Image Level): {ap_image}")
    print(f"AUROC (Pixel Level): {auroc_pixel}")
    print(f"AP (Pixel Level): {ap_pixel}")
    print(f"PRO (Pixel Level): {aupro}")
    metrics = {
        'AUROC-image':auroc_image,
        'AUROC-pixel':auroc_pixel,
        'AP-image':ap_image,
        'AP-pixel':ap_pixel,
        'AUPRO-pixel':aupro

    }
    
    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%% | AP-pixel: %.3f%% | AP-image: %.3f%%'  % 
                (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel'], metrics['AP-pixel'], metrics['AP-image']))


    return metrics
