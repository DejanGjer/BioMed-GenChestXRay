import os
import shutil
from argparse import ArgumentParser
import torch
import torchvision.utils as vutils
from generators import DCGANGenerator

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
if __name__=="__main__":
    """
    Example usage:
    python inference.py --ckpt_dir /home/mmilenkovic/git/BioMed-GenChestXRay/models/202407130030/ --num_images 5
    """
    ap=ArgumentParser()
    ap.add_argument("--ckpt_dir",type=str,help="path to load trained model from",required=True)
    ap.add_argument("--num_images",type=int,default=10)
    args=ap.parse_args()
    generator=DCGANGenerator.from_ckpt(ckpt_dir=args.ckpt_dir)    
    generator.to(DEVICE)
    z_batch=torch.randn((args.num_images,generator.z_size)).to(DEVICE)
    generated_img_batch = generator(z_batch).detach()
    img_save_dir=os.path.join(os.path.dirname(__file__),"generated_images.png")
    shutil.rmtree(img_save_dir, ignore_errors=True)
    vutils.save_image(generated_img_batch,img_save_dir,format="png")
