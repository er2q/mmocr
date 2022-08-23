log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT92', with_unknown=False, lower=True)
model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

img_norm_cfg = dict(mean=[127], std=[127])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'resize_shape', 'text', 'valid_ratio']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'resize_shape', 'valid_ratio', 'img_norm_cfg',
            'ori_filename', 'img_shape', 'ori_shape'
        ]),
]

train_root = '/disk_sda/wgh/dataset/ocr/SynthAdd'

train_img_prefix = f'{train_root}/SynthText_Add'
train_ann_file = f'{train_root}/label.lmdb'

train = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_list = [train]

test_root = 'data/mixture'

# test_img_prefix1 = f'{test_root}/IIIT5K/'
test_img_prefix1 = f'/disk_sda/wgh/dataset/ocr/textocr/'
# test_img_prefix2 = f'{test_root}/svt/'
# test_img_prefix3 = f'{test_root}/icdar_2013/'
# test_img_prefix4 = f'{test_root}/icdar_2015/'
# test_img_prefix5 = f'{test_root}/svtp/'
# test_img_prefix6 = f'{test_root}/ct80/'

# test_ann_file1 = f'{test_root}/IIIT5K/test_label.txt'
test_ann_file1 = f'/disk_sda/wgh/dataset/ocr/textocr/train_label.txt'
# test_ann_file2 = f'{test_root}/svt/test_label.txt'
# test_ann_file3 = f'{test_root}/icdar_2013/test_label_1015.txt'
# test_ann_file4 = f'{test_root}/icdar_2015/test_label.txt'
# test_ann_file5 = f'{test_root}/svtp/test_label.txt'
# test_ann_file6 = f'{test_root}/ct80/test_label.txt'

test1 = dict(
    type='OCRDataset',
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

# test2 = {key: value for key, value in test1.items()}
# test2['img_prefix'] = test_img_prefix2
# test2['ann_file'] = test_ann_file2
#
# test3 = {key: value for key, value in test1.items()}
# test3['img_prefix'] = test_img_prefix3
# test3['ann_file'] = test_ann_file3
#
# test4 = {key: value for key, value in test1.items()}
# test4['img_prefix'] = test_img_prefix4
# test4['ann_file'] = test_ann_file4
#
# test5 = {key: value for key, value in test1.items()}
# test5['img_prefix'] = test_img_prefix5
# test5['ann_file'] = test_ann_file5
#
# test6 = {key: value for key, value in test1.items()}
# test6['img_prefix'] = test_img_prefix6
# test6['ann_file'] = test_ann_file6

# test_list = [test1, test2, test3, test4, test5, test6]
test_list = [test1]

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5, max_keep_ckpts=10)
evaluation = dict(interval=5, metric='acc')

data = dict(
    samples_per_gpu=1280,
    workers_per_gpu=16,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))


cudnn_benchmark = True
work_dir = '/disk_sda/wgh/workplace/work_dirs/mmocr/seg_crnn_academic_dataset'
