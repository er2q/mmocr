# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp

import mmcv
import numpy as np

from mmocr.datasets.pipelines.crop import crop_img
from mmocr.utils.fileio import list_to_file, list_from_file


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list, imgs_list = [], []
    for img in os.listdir(img_dir):
        imgs_list.append(osp.join(img_dir, img))
        ann_list.append(osp.join(gt_dir, img.replace('jpg', 'txt')))

    files = list(zip(imgs_list, ann_list))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.join(osp.basename(img_file)),
        height=img.shape[0],
        width=img.shape[1],
        segm_file=osp.join(osp.basename(gt_file)))

    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_txt_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_txt_info(gt_file, img_info):
    """Collect the annotation information.

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    with open(gt_file, 'r', encoding='utf-8') as f:
        anno_info = []
        annotations = f.readlines()
        for ann in annotations:
            # Annotation format [x1, y1, x2, y2, x3, y3, x4, y4, transcript]
            try:
                bbox = np.array(ann.split(',')[0:8]).astype(int).tolist()
            except ValueError:
                # Skip invalid annotation line
                continue
            word = ann.split(',')[-1].replace('\n', '').strip()

            # Skip samples without recog gt
            if word == '###':
                continue
            anno = dict(bbox=bbox, word=word)
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def generate_ann(img_dir, image_infos, dst_image_root, json_path, keys_path, format='txt'):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
        format (str): Annotation format, should be either 'txt' or 'jsonl'
    """

    mmcv.mkdir_or_exist(dst_image_root)

    lines = []
    contents = ''
    if osp.exists(keys_path):
        for keys_line in list_from_file(keys_path):
            contents += keys_line

    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(img_dir, image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            dst_img = crop_img(image, anno['bbox'], 0, 0)
            h, w, _ = dst_img.shape

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            # Skip invalid annotations
            if min(dst_img.shape) == 0 or len(word) == 0:
                continue
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)

            contents += word
            if format == 'txt':
                lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                             f'{word}')
            elif format == 'jsonl':
                lines.append(
                    json.dumps({
                        'filename':
                            f'{osp.basename(dst_image_root)}/{dst_img_name}',
                        'text': word
                    }))
            else:
                raise NotImplementedError

    list_to_file(json_path, lines)
    list_to_file(keys_path, ''.join(set(contents)))


def data_transfor(img_dir, gt_dir, crop_path, json_path, keys_path):
    with mmcv.Timer(print_tmpl='It takes {}s to convert DeText annotation'):
        files = collect_files(img_dir, gt_dir)
        image_infos = collect_annotations(files, nproc=1)
        generate_ann(img_dir, image_infos, crop_path, json_path, keys_path)


if __name__ == '__main__':
    image_dir = 'E:/datasets/ocr/OCR_dataset_test/imgs'
    label_dir = 'E:/datasets/ocr/OCR_dataset_test/annotations'
    crop_path = 'E:/datasets/ocr/OCR_dataset_test/crop'
    json_path = 'E:/datasets/ocr/OCR_dataset_test/train_label.txt'
    keys_path = 'E:/datasets/ocr/OCR_dataset_test/keys.txt'

    data_transfor(image_dir, label_dir, crop_path, json_path, keys_path)
