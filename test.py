# -*- coding:UTF-8 -*-

# @author  : admin
# @file    : test.py
# @datetime: 2022/8/24 10:33
# @software: PyCharm

"""
文件说明：模型大小
    
"""
import os.path as osp
from glob import glob
from mmcv import Config

from mmocr.models import build_detector


def get_model_size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    # buffer_size = 0
    # buffer_sum = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #     buffer_sum += buffer.nelement()
    all_size = (param_size) / 1024 / 1024       # K,MB
    print('模型总大小为：{:.3f}MB'.format(all_size))
    # return param_size, param_sum, buffer_size, buffer_sum, all_size


if __name__ == '__main__':
    for config_path in glob(osp.join('configs/textdet', '*/*.py')):
        print(config_path)
        cfg = Config.fromfile(config_path)
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        get_model_size(model)
        # total = sum([param.nelement() for param in model.parameters()])
        #
        # print("Number of parameter: %.2fM" % (total / 1024 / 1024))
