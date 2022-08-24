# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import sys
import os.path as osp
import time
import warnings
import mmcv
import torch
from mmcv import Config
from mmcv.utils import get_git_hash
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from PySide2.QtCore import Signal, QThread

# 临时环境变量
BASE_DIR = osp.dirname(osp.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(osp.dirname(BASE_DIR))

from utils.general import get_logger
# from utils.qt_utils import CatchedException
from utils.ocr_data_transformation_seg import data_transfor

from mmocr import __version__
from mmocr.apis import train_detector
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.utils import is_2dlist

CONFIG_DIR = osp.join(osp.dirname(__file__), 'configs')
dict_config_path = {
    'CRNN': osp.join(CONFIG_DIR, 'seg_crnn_icdar2017_RCTW.py'),
}


class Train(QThread):
    state_ = Signal(str)

    def __init__(self, image_dir, label_dir, load_from, learning_rate, batch_size, max_epochs, work_dir, network):
        super(Train, self).__init__()
        self.network = network  # 网络模型结构
        self.image_dir = image_dir  # 图像文件夹
        self.label_dir = label_dir  # 标签文件夹
        # self.num_classes = num_classes  # 类别数
        self.learning_rate = learning_rate  # 学习率
        if batch_size > 1 and batch_size % 2 == 1:  # qt 界面设置最小值为 1
            batch_size -= 1  # 向下取偶数
        self.batch_size = batch_size  # 单次训练数量
        self.max_epochs = max_epochs  # 训练轮次
        self.work_dir = work_dir  # 模型保存文件夹
        self.load_from = load_from  # 预训练模型
        self.resume_from = None
        self.gpu_ids = None  # 指定使用 gpu，用列表
        self.launcher = 'none'  # 使用分布式
        self.seed = None
        self.deterministic = False

    def data_transformation(self, split):
        """
        将数据转换为detect训练格式
        :return:
        """
        out_path = osp.join(osp.dirname(self.image_dir), 'instances_' + split + '.json')
        data_transfor(osp.join(self.image_dir, split), osp.join(self.label_dir, split), out_path)
        data_transfor(self.image_dir, self.label_dir, out_path)
        return out_path

    def run(self):
        if not torch.cuda.is_available():
            self.state_.emit('当前设备 GPU 不可用，无法训练。')
            # raise CatchedException

        config_path = dict_config_path[self.network]
        try:
            cfg = Config.fromfile(config_path)
        except:
            self.state_.emit('解析配置文件失败！配置文件不存在或被修改。')
            # raise CatchedException

        cfg.data.samples_per_gpu = self.batch_size
        cfg.data.workers_per_gpu = 1  # worker 不能 == 0
        """
        标准格式:
            - image_dir
                - train
                - val
            - label_dir
                - train
                - val
        """
        cfg.data.train.datasets[0].img_prefix = self.image_dir
        cfg.data.val.datasets[0].img_prefix = self.image_dir
        if osp.exists(osp.join(self.image_dir, 'train')) and osp.exists(osp.join(self.image_dir, 'val')) and osp.exists(
                osp.join(self.label_dir, 'train')) and osp.exists(osp.join(self.label_dir, 'val')):
            cfg.data.train.datasets[0].ann_file = self.data_transformation('train')
            cfg.data.val.datasets[0].ann_file = self.data_transformation('val')
        else:
            try:
                assert not (osp.exists(osp.join(self.image_dir, 'train')) or osp.exists(
                    osp.join(self.image_dir, 'val')) or osp.exists(
                    osp.join(self.label_dir, 'train')) or osp.exists(osp.join(self.label_dir, 'val'))), '训练数据集目录格式错误'
            except:
                self.state_.emit('训练数据集目录格式错误。')
                # raise CatchedException
            ann_file = self.data_transformation('')
            cfg.data.train.datasets[0].ann_file = ann_file
            cfg.data.val.datasets[0].ann_file = ann_file
        cfg.optimizer.lr = self.learning_rate
        cfg.runner.max_epochs = self.max_epochs

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.work_dir = self.work_dir if self.work_dir else f'work_dir/{self.network}'

        cfg.load_from = self.load_from
        if self.resume_from:
            assert osp.exists(self.resume_from)
            cfg.resume_from = self.resume_from
        if self.gpu_ids:
            cfg.gpu_ids = self.gpu_ids
        else:
            cfg.gpu_ids = range(1)
        if self.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(self.launcher, **cfg.dist_params)
            # gpu_ids is used to calculate iter when resuming checkpoint
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)
        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump configs
        cfg.dump(osp.join(cfg.work_dir, osp.basename(config_path)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        logger = get_logger(log_name='train')  # 创建训练日志
        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()

        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        if self.seed is not None:
            logger.info(f'Set random seed to {self.seed}, deterministic: '
                        f'{self.deterministic}')
            set_random_seed(self.seed, deterministic=self.deterministic)
        cfg.seed = self.seed
        meta['seed'] = self.seed
        meta['exp_name'] = osp.basename(config_path)

        try:
            model = build_detector(
                cfg.model,
                train_cfg=cfg.get('train_cfg'),
                test_cfg=cfg.get('test_cfg'))
            model.init_weights()
        except:
            self.state_.emit('模型初始化失败！配置文件解析错误。')
            # raise CatchedException

        # SyncBN is not support for DP
        if not distributed:
            warnings.warn(
                'SyncBN is only supported with DDP. To be compatible with DP, '
                'we convert SyncBN to BN. Please use dist_train.sh which can '
                'avoid this error.')
            model = revert_sync_batchnorm(model)

        logger.info(model)
        try:
            datasets = [build_dataset(cfg.data.train)]
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                if cfg.data.train.get('pipeline', None) is None:
                    if is_2dlist(cfg.data.train.datasets):
                        train_pipeline = cfg.data.train.datasets[0][0].pipeline
                    else:
                        train_pipeline = cfg.data.train.datasets[0].pipeline
                elif is_2dlist(cfg.data.train.pipeline):
                    train_pipeline = cfg.data.train.pipeline[0]
                else:
                    train_pipeline = cfg.data.train.pipeline

                if val_dataset['type'] in ['ConcatDataset', 'UniformConcatDataset']:
                    for dataset in val_dataset['datasets']:
                        dataset.pipeline = train_pipeline
                else:
                    val_dataset.pipeline = train_pipeline
                datasets.append(build_dataset(val_dataset))
        except Exception as e:
            self.state_.emit('加载数据集失败！' + str(e))
            # raise CatchedException

        if cfg.checkpoint_config is not None:
            # save mmseg version, configs file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmocr_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
        # passing checkpoint meta for saving best checkpoint
        meta.update(cfg.checkpoint_config.meta)
        try:
            train_detector(
                model,
                datasets,
                cfg,
                distributed=distributed,
                validate=True,
                timestamp=timestamp,
                meta=meta)
        except RuntimeError as e:
            if 'CUDA out of memory.' in str(e):
                self.state_.emit('CUDA内存溢出，尝试减小 batch size，或者选择更小的模型。')
            # raise CatchedException
        except ValueError as e:
            if 'Expected more than 1 value per channel when training' in str(e):
                self.state_.emit('该模型需要的 batch size 最小为2。')
                # raise CatchedException
        logging.shutdown()


if __name__ == '__main__':
    image_dir = 'E:/datasets/ocr/OCR_dataset_test/imgs'
    label_dir = 'E:/datasets/ocr/OCR_dataset_test/annotations'
    load_from = None
    learning_rate = 0.007
    batch_size = 20
    max_epochs = 10
    work_dir = 'E:/datasets/ocr/OCR_dataset_test/works_dir'
    network = 'CRNN'
    Train(image_dir, label_dir, load_from, learning_rate, batch_size, max_epochs, work_dir, network).run()
