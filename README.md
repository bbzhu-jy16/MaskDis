# High-quality and Diverse Few-shot Image Generation via Masked Discrimination

### Requirements
- Linux
- NVIDIA GPU (NVIDIA TITAN RTX in our experiments) + CUDA CuDNN 11.2
- PyTorch 1.7.0
- torchvision 0.8.1
- Python 3.6.9
- Install all the required libraries:
         `pip install -r requirements.txt` 

### Training and Evaluation

#### Prepare datasets

##### Transfer target images to lmdb format for GAN adaptation.

`CUDA_VISIBLE_DEVICES='0' python prepare_data.py --out path/to/datasets --size 256 path/to/original/images`

##### Prepare folders for Intra-LPIPS evaluation (following CDC's work)

#### We provide a folder structure example of Amedeo's paintings in our code for Intra-LPIPS evaluation. 

#### Train the adapted model and evaluate Intra-LPIPS 

1.Checkpoints and samples are saved in checkpoints/exp_name and samples/exp_name automatically.

2.We implement codes to evaluate Intra-LPIPS with fixed input noise vectors and print results in training process. 

3.The noise vectors used for our experiments are provided in "test_noise.pt".

`CUDA_VISIBLE_DEVICES='0' python train.py --ckpt path/to/source/model --data_path path/to/datasets  --exp <exp_name> --dataset <dataset_name>`

### FID evaluation

We follow prior works to use pytorch-fid for FID evaluation (carried out for abundant datasets).

#### Install pytorch-fid through pip first:

`pip install pytorch-fid==0.1.1`

#### Generate fake images for FID evaluation:

`CUDA_VISIBLE_DEVICES='0' python generate.py --ckpt_target /path/to/model/ --imsave_path /path/to/fake/images`

#### Calculate FID 

`CUDA_VISIBLE_DEVICES='0' python -m pytorch_fid /path/to/real/images /path/to/fake/images`

#### Citation






