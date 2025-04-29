import os
import pathlib
import torch
import cv2
from tqdm import tqdm

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
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
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

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, short_size: int = 1024):
        assert os.path.exists(img_path), 'file does not exist'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img, original_size, resized_size = resize_image(img, short_size)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        batch = {'shape': [original_shape]}
        with torch.no_grad():
            preds = self.model(tensor)
            preds = preds[0, 0, :, :].detach().cpu().numpy()
        # Resize predictions back to original size
        preds_resized = cv2.resize(preds, (original_size[0], original_size[1]))
        return preds_resized


def save_guidance(preds, output_path):
    cv2.imwrite(output_path, preds * 255)


if __name__ == '__main__':
    model_path = '/data1/cxy/model_besttd.pth'
    input_folder = '/data1/cxy/DBNET/datasets1/test/img'
    output_folder = '/data1/cxy/guide200'
    os.makedirs(output_folder, exist_ok=True)

    model = PytorchModel(model_path, post_p_thre=0.3, gpu_id=0)

    for img_path in tqdm(list(pathlib.Path(input_folder).glob('*.JPG'))):
        preds = model.predict(str(img_path))
        output_path = os.path.join(output_folder, img_path.stem + '.JPG')
        save_guidance(preds, output_path)
