import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import get_model
import torchvision.transforms as T
from datasets.CamVid_dataloader11 import *



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./datasets/test', help='Input image or folder')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/segnet_best.pth', help='Checkpoint path')
    parser.add_argument('--model', type=str, default='segnet', help='Segmentation head')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes')
    parser.add_argument('--save_dir', type=str, default='./predictions', help='Directory to save results')
    parser.add_argument('--overlay', type=bool, default=True, help='Save overlay image')
    return parser.parse_args()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), image  # tensor, PIL image

#把类别mask ➔ 彩色图 (用VOC_COLORMAP)
def mask_to_color(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in range(len(Cam_COLORMAP)):
        color_mask[mask == label] = Cam_COLORMAP[label]
    return color_mask

def save_mask(mask, save_path):
    color_mask = mask_to_color(mask)
    Image.fromarray(color_mask).save(save_path)

def overlay_mask_on_image(raw_image, mask, alpha=0.6):
    mask_color = mask_to_color(mask)
    mask_pil = Image.fromarray(mask_color)
    mask_pil = mask_pil.resize(raw_image.size, resample=Image.NEAREST)
    blended = Image.blend(raw_image, mask_pil, alpha=alpha)
    return blended

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 模型
    model = get_model(num_classes=args.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    # 预测单张 or 批量
    if os.path.isdir(args.image_dir):
        image_list = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    else:
        image_list = [args.image]

    print(f"🔎 Found {len(image_list)} images to predict.")

    for img_path in tqdm(image_list):
        img_tensor, raw_img = load_image(img_path)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # 保存 mask
        base_name = os.path.basename(img_path).split('.')[0]
        mask_save_path = os.path.join(args.save_dir, f"{base_name}_mask.png")
        save_mask(pred, mask_save_path)

        # 保存 overlay
        if args.overlay:
            overlay_img = overlay_mask_on_image(raw_img, pred)
            overlay_save_path = os.path.join(args.save_dir, f"{base_name}_overlay.png")
            overlay_img.save(overlay_save_path)

        print(f"Saved: {mask_save_path}")
        if args.overlay:
            print(f"Saved overlay: {overlay_save_path}")

    print("🎉 Prediction done!")

if __name__ == '__main__':
    args = parse_arguments()
    predict(args)
