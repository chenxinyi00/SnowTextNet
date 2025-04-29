# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 13:12
# @Author  : zhoujun
import copy
import cv2
import numpy as np
from torch.utils.data import Dataset
from data_loader.modules import *


class BaseDataSet(Dataset):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags, transform=None, target_transform=None):
        assert img_mode in ['RGB', 'BGR', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)  # ✅ 读取成对的雪图和去雪图
        item_keys = ['snow_img_path', 'img_path', 'img_name','text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], f'data_list from load_data 必须包含 {item_keys}'

        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        读取数据，确保雪图和去雪图是成对的
        :param data_path: 数据存储路径
        :return: 返回包含成对图像信息的列表
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        """
        数据增强同时作用于 snow_img 和 img
        """
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        try:
            data = copy.deepcopy(self.data_list[index])

            # ✅ 读取雪图
            img = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
            snow_img = cv2.imread(data['snow_img_path'], 1 if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                snow_img = cv2.cvtColor(snow_img, cv2.COLOR_BGR2RGB)


            # ✅ 存入字典
            data['snow_img'] = snow_img
            data['img'] = img
            data['shape'] = [snow_img.shape[0], snow_img.shape[1]]
            data = self.apply_pre_processes(data)

            # ✅ 数据增强同时作用于 snow_img 和 img
            # augmented = self.apply_pre_processes({
            #     'snow_img': data['snow_img'],
            #     'img': data['img'],
            #     'text_polys': data['text_polys'],
            #     'texts': data['texts'],
            #     'ignore_tags': data['ignore_tags'],
            # })
            # data.update(augmented)

            # ✅ 变换同时作用于 snow_img 和 img
            if self.transform:
                data['snow_img'] = self.transform(data['snow_img'])
                data['img'] = self.transform(data['img'])

            # ✅ 处理 `text_polys`
            data['text_polys'] = data['text_polys'].tolist()

            # ✅ 过滤不需要的 key
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                return data_dict
            else:
                return data

        except Exception as e:
            print(f"数据加载错误: {e}，随机选取新样本")
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


