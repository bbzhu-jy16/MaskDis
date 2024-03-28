import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
import os

def resize_multiple(
    img, image_num, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    for size in sizes:
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
        img.save('datasets/sketches/c'+str(image_num)+'/center.png', format="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("path", type=str, help="path to the image dataset")
    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]
    
    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))
    
    images = os.listdir(args.path)
    
    image_num = -1
    for image in images:
        image_num+=1
        img = Image.open(args.path+'/'+image)
        img = img.convert("RGB")
        out = resize_multiple(img, sizes=sizes, resample=resample, quality=100, image_num=image_num)