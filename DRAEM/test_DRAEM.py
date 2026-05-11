
import os
import torch
import torch.nn.functional as F
import cv2
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from au_pro_util import calculate_au_pro

def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)

def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

# def test(obj_names, mvtec_path, checkpoint_path, base_model_name,Dataset_name,percent,visualization_path):
def test(obj_names, mvtec_path, checkpoint_path, base_model_name,Dataset_name,percent):
    print(obj_names, mvtec_path, checkpoint_path, base_model_name,Dataset_name,percent)
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    obj_pro_image_list = []
    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name+"_"+obj_name+'_'
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()
        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []
        gt_masks=[] #
        predicted_masks=[] #

        display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
        display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
        cnt_display = 0
        display_indices = np.random.randint(len(dataloader), size=(16,))

        # if(not os.path.exists(visualization_path+'/'+obj_name)):
        #         os.makedirs(visualization_path+'/'+obj_name)
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            image_path = sample_batched["image_path"] # 
            
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            # anomaly_map_cur = out_mask_sm[0,1,:].cpu().detach().numpy()*255
            # heatmap = cv2heatmap(anomaly_map_cur)
            # # print(image_path[0])
            # cv2.imwrite(os.path.join(visualization_path,obj_name,str(i_batch)+'_anomaly_map.png'),anomaly_map_cur)
            # cv2.imwrite(os.path.join(visualization_path,obj_name,str(i_batch)+'_heatmap.png'),heatmap)
            # origin_image = cv2.imread(image_path[0])
            # origin_image = cv2.resize(origin_image, (256, 256))
            # heatmap_on_img = heatmap_on_image(heatmap, origin_image)
            # cv2.imwrite(os.path.join(visualization_path,obj_name,str(i_batch)+'_origin_image.png'),origin_image)
            # cv2.imwrite(os.path.join(visualization_path,obj_name,str(i_batch)+'_heatmap_on_img.png'),heatmap_on_img)
            # mask1=true_mask.squeeze(0)
            # mask2=mask1.squeeze(0)
            # cv2.imwrite(os.path.join(visualization_path,obj_name,str(i_batch)+'_gt.png'),mask2.cpu().numpy()*255)

            if i_batch in display_indices:
                t_mask = out_mask_sm[:, 1:, :, :]
                display_images[cnt_display] = gray_rec[0]
                display_gt_images[cnt_display] = gray_batch[0]
                display_out_masks[cnt_display] = t_mask[0]
                display_in_masks[cnt_display] = true_mask[0]
                cnt_display += 1


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

        total_gt_pixel_scores_pro = np.array(total_gt_pixel_scores).reshape(len(dataset),img_dim,img_dim)
#         print(total_gt_pixel_scores_pro.shape)
        total_pixel_scores_pro = np.array(total_pixel_scores).reshape(len(dataset),img_dim,img_dim)
#         print(total_pixel_scores_pro.shape)
        pro_pixel ,_ = calculate_au_pro(total_gt_pixel_scores_pro, total_pixel_scores_pro) #

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        obj_pro_image_list.append(pro_pixel) # 
        print(obj_name)
        print("AUC Image:  " +str(auroc))        
        print("AUC Pixel:  " +str(auroc_pixel))
        print("AP Image:  " +str(ap))

        print("AP Pixel:  " +str(ap_pixel))
        print("Pro Pixel: "+str(pro_pixel))
        print("==============================")

#     print(run_name)
#     print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
#     print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
#     print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
#     print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=False)
    parser.add_argument('--base_model_name', action='store', type=str, required=False,default='DRAEM_test_0.0001_500_bs8')
    parser.add_argument('--data_path', action='store', type=str, default='/cluster/home/zqyeleven/ASBenchmark/BTAD/')
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    # parser.add_argument('--visualization_path', action='store', type=str, required=True)
    parser.add_argument('--Dataset_name',type=str, required=True)
    parser.add_argument('--percent',type=str,required=True)

    args = parser.parse_args()

    obj_list = [
                'bottle',
                'capsule',
                'carpet',
                'cable',
                'leather',
                'pill',
                'transistor',
                'tile',
                # 'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'screw',
                # 'wood',
                'grid'
                 ]

    with torch.cuda.device(args.gpu_id):
        test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name,args.Dataset_name,args.percent)
        # test(obj_list,args.data_path, args.checkpoint_path, args.base_model_name,args.Dataset_name,args.percent,args.visualization_path)


