# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : 6.文本检测+识别.py
# @datetime: 2022/8/21 17:40
# @software: PyCharm

"""
文件说明：
    
"""
import os
import os.path as osp
# import cv2
# cv2.imencode()
from mmocr.utils.ocr import MMOCR

# 导入模型到内存
# ocr = MMOCR('DB_r18', '/disk_sda/wgh/workplace/mmocr/myconfigs/dbnet_r18_fpnc_1200e_icdar2017_RCTW.py',
#             '/disk_sda/wgh/workplace/work_dirs/mmocr/dbnet_r18_fpnc_1200e_icdar2017_RCTW/latest.pth',
#             'CRNN', '/disk_sda/wgh/workplace/mmocr/myconfigs/seg_crnn_icdar2017_RCTW.py',
#             '/disk_sda/wgh/workplace/work_dirs/mmocr/seg_crnn_academic_dataset/latest.pth')

ocr = MMOCR()

# 推理
src = '/disk_sda/wgh/dataset/ocr/ICDAR2017_RCTW/test_images'
dst_output = '/disk_sda/wgh/dataset/ocr/ICDAR2017_RCTW/test_images_result1'
dst_export = '/disk_sda/wgh/dataset/ocr/ICDAR2017_RCTW/test_images_result2'
os.makedirs(dst_export, exist_ok=True)
for img_name in os.listdir(src):
    results = ocr.readtext(osp.join(src, img_name), output=osp.join(dst_output, img_name), details=True,
                           export=dst_export, print_result=True, merge=False)
