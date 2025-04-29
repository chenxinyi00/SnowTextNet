import argparse
import os
import torch
import numpy as np
import pathlib
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from metrics import SSIM, PSNR
from dataloader import CSD_Dataset, SRRS_Dataset, Snow100K_Dataset, Text, Text1, TextTD
from DGSNet import SnowFormer
from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing

def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (width, height), (new_width, new_height)

class PytorchModel:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms([
            t for t in config['dataset']['train']['dataset']['args']['transforms'] if t['type'] in ['ToTensor', 'Normalize']
        ])

    def predict(self, img_path: str, short_size: int = 1024):
        assert os.path.exists(img_path), 'file does not exist'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img, original_size, resized_size = resize_image(img, short_size)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)[0, 0, :, :].detach().cpu().numpy()
        return cv2.resize(preds, (original_size[0], original_size[1]))

def save_guidance(preds, output_path):
    cv2.imwrite(output_path, preds * 255)

def generate_guidance_images(model, dataset_path):
    guide_folder = os.path.join(dataset_path, 'guide')
    os.makedirs(guide_folder, exist_ok=True)
    for img_path in tqdm(list(pathlib.Path(dataset_path).glob('*.JPG'))):
        output_path = os.path.join(guide_folder, img_path.name)
        if not os.path.exists(output_path):
            preds = model.predict(str(img_path))
            save_guidance(preds, output_path)
    return guide_folder

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='TextTD', help='CSD/SRRS/Snow100K/Text')
parser.add_argument('--dataset_path', type=str, default='/data1/cxy/TextTD/', help='Dataset path')
parser.add_argument('--savepath', type=str, default='/data1/cxy/outtd/', help='Output path')
parser.add_argument('--guidance_model_path', type=str, default='/data1/cxy/model_besttd.pth', help='Guidance model path')
parser.add_argument('--snowformer_model_path', type=str, default=' /data1/cxy/checkpoints3/SnowFormer_epoch999.pth', help='SnowFormer model path')
parser.add_argument('--cuda', action='store_true', help='Use GPU')
opt = parser.parse_args()

# 生成 Guidance 图片
guidance_model = PytorchModel(opt.guidance_model_path, post_p_thre=0.3, gpu_id=0)
guide_folder = generate_guidance_images(guidance_model, opt.dataset_path)

# 选择数据集
data_loader_mapping = {
    'CSD': CSD_Dataset,
    'SRRS': SRRS_Dataset,
    'Snow100K': Snow100K_Dataset,
    'Text': Text,
    'Text1': Text1,
    'TextTD': TextTD
}
snow_test = DataLoader(
    dataset=data_loader_mapping[opt.dataset_type](opt.dataset_path, train=False, size=256, rand_inpaint=False, rand_augment=None),
    batch_size=1, shuffle=False, num_workers=4)

# 初始化 SnowFormer 模型
netG_1 = SnowFormer().cuda()
netG_1.load_state_dict(torch.load(opt.snowformer_model_path))

savepath_dataset = os.path.join(opt.savepath, opt.dataset_type)
os.makedirs(savepath_dataset, exist_ok=True)

ssims, psnrs = [], []
loop = tqdm(enumerate(snow_test), total=len(snow_test))

for idx, (haze, clean, _, name) in loop:
    guidance_path = os.path.join(guide_folder, name[0] + '.JPG')
    if not os.path.exists(guidance_path):
        print(f"Warning: Guidance image {guidance_path} not found!")
        continue
    guidance = torch.tensor(cv2.imread(guidance_path, 0)).unsqueeze(0).cuda()
    with torch.no_grad():
        haze, clean = haze.cuda(), clean.cuda()
        dehaze = netG_1(haze, guidance)
    save_image(dehaze, os.path.join(savepath_dataset, f'{name[0]}.jpg'))
    ssims.append(SSIM(dehaze, clean).item())
    psnrs.append(PSNR(dehaze, clean))
    print(f'Generated images {idx + 1} of {len(snow_test)}, SSIM: {ssims[-1]}, PSNR: {psnrs[-1]}')

print(f'Average SSIM: {np.mean(ssims)}, Average PSNR: {np.mean(psnrs)}')
