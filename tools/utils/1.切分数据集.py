# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : 1.切分数据集.py
# @datetime: 2022/8/20 11:52
# @software: PyCharm

"""
文件说明：切分数据集
    
"""
import os
import shutil
import os.path as osp
from tqdm import tqdm


def split_images_masks_to_train_val(data_root, image_suffix, mask_suffix, split_rate=0.2):
    """
    划分训练集和验证集
    :param data_root: 图片所在目录
    :param image_suffix: 图片后缀
    :param mask_suffix: 标签图片后缀
    :param split_rate: 比例
    :return:
    """
    image_dir = osp.join(data_root, 'imgs')
    mask_dir = osp.join(data_root, 'annotations')

    img_train_dir = osp.join(image_dir, 'train')
    img_val_dir = osp.join(image_dir, 'val')
    ann_train_dir = osp.join(mask_dir, 'train')
    ann_val_dir = osp.join(mask_dir, 'val')

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(ann_train_dir, exist_ok=True)
    os.makedirs(ann_val_dir, exist_ok=True)

    imgs_list = os.listdir(image_dir)
    anns_list = os.listdir(mask_dir)
    assert len(imgs_list) == len(anns_list), "图片数量不等于标签数量"

    split_interval = int(1 / split_rate)

    for index, file in enumerate(imgs_list):
        file1_path = osp.join(image_dir, file)
        file2_path = osp.join(mask_dir, file.replace(image_suffix, mask_suffix))
        if index % split_interval == 0:  # 验证集
            if file1_path.endswith(image_suffix):
                shutil.move(file1_path, img_val_dir)
            if file2_path.endswith(mask_suffix):
                shutil.move(file2_path, ann_val_dir)
            continue
        # 训练集
        if file1_path.endswith(image_suffix):
            shutil.move(file1_path, img_train_dir)
        if file2_path.endswith(mask_suffix):
            shutil.move(file2_path, ann_train_dir)
        continue


if __name__ == '__main__':
    src_dir = '/disk_sda/wgh/dataset/ocr/OCR数据集格式演示'
    split_images_masks_to_train_val(src_dir, 'jpg', 'txt')
