# ASBench
  
ASBench: Image Anomalies Synthesis Benchmark for Anomaly Detection [[IEEE Transactions on Artificial Intelligence]](https://arxiv.org/abs/2510.07927)


## Data preparation 
Download the following datasets:
- **MVTec-AD [[Official]](https://www.mvtec.com/company/research/datasets/mvtec-ad/)**  
- **MPDD [[Official]](https://github.com/stepanje/mpdd)**  
- **BTAD [[Official]](http://avires.dimi.uniud.it/papers/btad/btad.zip)**  
- **VisA [[Official]](https://github.com/amazon-science/spot-diff)**  
- **MTD [[Official]](https://github.com/abin24/Magnetic-tile-defect-datasets)**

- **DTD (optional) dataset for anomaly synthesis [[Official]](https://www.robots.ox.ac.uk/~vgg/data/dtd/)**  
  
For specific usage instructions, please refer to the steps of each pipelines.

## DestSeg
This part is based on the following link: **DestSeg README [[Link]](https://github.com/apple/ml-destseg/blob/main/README.md)**
### Installation
pip install -r destseg/requirements.txt
### Datasets

Take the MVTec AD datasets as an example. Users can run the **download_dataset.sh** script to download them directly.

```
.destseg/scripts/download_dataset.sh
```

### Training and Testing

To get started, users can run the following command to train the model on all categories of MVTec AD dataset:

```
python train.py \
    --gpu_id 0 \
    --num_workers 16 \
    --custom_training_category \
    --no_rotation_category bottle \
    --checkpoint_path saved_model/ \
    --mvtec_path ./destseg/datasets/mvtec/ \
    --Dataset DestSeg \
    --percent 1 \
```
The above are the basic commands. "Dataset" represents the name of the synthesis method, with options including "AD, CutOut, CutPaste, CutPaste_Scar, DestSeg, DFMGAN, DRAEM, FPI, Fractal, MemSeg, NSA, RealNet."
"Percent" indicates the proportion of anomalous samples. When set to 1, all samples are processed as anomalous samples.

If normal samples, synthesized anomalous samples, and masks are already prepared, they can be invoked using the following:
```
    --aug_dir xxx \
    --mask_dir xxx \
    --origin_dir xxx \
```

To test the performance of the model, users can run the following command:

```
python eval.py --gpu_id 0 --num_workers 16
```


## DRAEM
This part is based on the following link: **DRAEM README [[Link]](https://github.com/VitjanZ/DRAEM/blob/main/README.md)**


### Installation
The conda environement used in the project is decsribed in **DRAEM/requirements.txt**.

### Datasets
Take the MVTec AD datasets as an example. Users can run the **download_dataset.sh** script to download them directly.

```
.destseg/scripts/download_dataset.sh
```

### Training
Pass the folder containing the training dataset to the **train_DRAEM.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument. 
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path) and path to store logs (--log_path).
Example:

```
 CUDA_VISIBLE_DEVICES=0 python test_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --base_model_name "DRAEM_test_0.0001_700_bs8" --data_path ./DRAEM/datasets/mvtec/ --checkpoint_path ./DRAEM/checkpoints/  --log_path ./DRAEM/logs/ --Dataset_name DRAEM --percent 0.5
```


### Testing
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the 
location of the MVTec anomaly detection dataset (--data_path) and the folder where the checkpoint files are located (--checkpoint_path)
with pretrained models can be run with:

```
python test_DRAEM.py --gpu_id 1 --base_model_name "DRAEM_test_0.0001_700_bs8" --data_path ./DRAEM/datasets/mvtec/ --checkpoint_path ./DRAEM/checkpoints/
```


## MemSeg
This part is based on the following link: **DRAEM README [[Link]](https://github.com/VitjanZ/DRAEM/blob/main/README.md)**

 

### Installation

- Docker image: nvcr.io/nvidia/pytorch:20.12-py3

```
einops==0.5.0
timm==0.5.4
wandb==0.12.17
omegaconf
imgaug==0.4.0
```


### Training

```bash
python main.py configs=configs.yaml EXP_NAME=MemSeg DATASET.target=bottle DATASETNAME=MemSeg DATASET.percent=0.5 RESULT.savedir=./saved_model/  DATASET.datadir=/mvtec/
```
### Testing
```bash
python main_eval.py configs=configs.yaml EXP_NAME=MemSeg DATASET.target=bottle DATASETNAME=MemSeg DATASET.percent=0.5 RESULT.savedir=./saved_model/  DATASET.datadir=/mvtec/
```

## anomalydiffusion
Since the code reproduction process of AnomalyDiffusion itself involves saving and then reading the generated images, here we have adopted the approach of saving and then reading the images for all generation methods.
### Installation
```
Ubuntu
python 3.8
cuda==11.8
gcc==7.5.0
conda env create -f environment.yaml
conda activate Anomalydiffusion
```

### Training
After generating anomalous image-mask pairs,
you can train and test the **anomaly detection model** (for both anomlay detection and localization) by:
```
python train-localization.py --generated_data_path $path_to_the_generated_data  --mvtec_path=$path_to_mvtec --percent 0.5 --Dataset_name AD
```

### Testing

```
python test-localization.py --generated_data_path $path_to_mvtec
```
