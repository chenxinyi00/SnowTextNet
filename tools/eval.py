# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import sys
import pathlib
import argparse
import time
import torch
from tqdm.auto import tqdm

# 当前路径设置
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

class EVAL():
    def __init__(self, model_path, gpu_id=0, distributed=False):
        from models import build_model
        from data_loader import get_dataloader
        from post_processing import get_post_processing
        from utils import get_metric

        # 检查是否使用 GPU
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        # 加载模型权重和配置
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        # 强制覆盖分布式配置
        config['distributed'] = distributed

        # 加载数据
        self.validate_loader = get_dataloader(config['dataset']['validate'], distributed=config['distributed'])

        # 构建模型
        self.model = build_model(config['arch'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

        # 后处理和评测指标
        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0

        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据转换并丢到 GPU/CPU
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)

                # 计时开始
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(
                    batch, preds, is_output_polygon=self.metric_cls.is_output_polygon
                )
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start

                # 计算指标
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)

        # 汇总评测指标
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', required=False, default='/data1/cxy/model_latest.pth', type=str)
    parser.add_argument('--distributed', action='store_true', help='Enable distributed mode')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (default: 0)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()

    # 初始化分布式环境（如果启用分布式模式）
    if args.distributed:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://')

    # 实例化评测器
    eval = EVAL(args.model_path, gpu_id=args.gpu_id, distributed=args.distributed)
    result = eval.eval()

    # 打印评测结果
    print("Recall: {:.4f}, Precision: {:.4f}, F-measure: {:.4f}".format(*result))
