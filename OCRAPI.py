# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : 6.文本检测+识别.py
# @datetime: 2022/8/21 17:40
# @software: PyCharm

"""
文件说明：

"""
import os, sys
from mmocr.utils.ocr import MMOCR


def ocr_run(file_name, img_path, det=None, recog=None, dst_export=None):
    """

    :param file_name: 文件名，无用
    :param img_path: 图片路径
    :param det: 文字检测模型路径
    :param recog: 文本识别模型路径
    :param dst_export: json保存路径
    :return:
    """

    # 是否传入模型
    if det and recog:
        ocr = MMOCR(
            'DB_r18', '/disk_sda/wgh/workplace/mmocr/myconfigs/dbnet_r18_fpnc_1200e_icdar2017_RCTW.py', det,
            'CRNN', '/disk_sda/wgh/workplace/mmocr/myconfigs/seg_crnn_icdar2017_RCTW.py', recog
        )
    else:
        ocr = MMOCR()

    filepath, filename = os.path.split(img_path)
    if not dst_export:
        dst_export = filepath
    ocr.readtext(img_path, details=True, export=dst_export, print_result=True, merge=False)


if __name__ == '__main__':
    # img_path = 'C:/Users/admin/Desktop/orc_test.jpg'
    # print(sys.argv)

    ocr_run(*sys.argv)
