from lightning_model import TextClassificationLightningModel
import argparse
from PIL import Image
import numpy as np
from dataset import LetterBox
from torchvision import transforms
import torch
from model import mobileone

parser = argparse.ArgumentParser(description='Bearing Faults Project Configuration')
parser.add_argument('--checkpoint', type=str, help='checkpoint path')
parser.add_argument('--img', type=str, help='path of image')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()


def main():
    transform = transforms.Compose([
            LetterBox(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # Load image from path
    image_raw = Image.open(args.img).convert('RGB')
    image_raw = np.array(image_raw)
    image_pil = Image.fromarray(image_raw)
    image_tensor = transform(image_pil).unsqueeze(dim=0).to(args.device)

    # Load model from checkpoint
    net = mobileone(num_classes=4, variant='s1')
    model = TextClassificationLightningModel.load_from_checkpoint(args.checkpoint, model=net)
    
    "inference phase"
    prediction = model.predict(image_tensor)
    print(prediction)
if __name__ == "__main__":
    main()