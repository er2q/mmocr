# -*- coding: utf-8 -*-
import os, sys
import os.path as osp
import random
import shutil
import numpy as np
from tqdm import tqdm
import codecs
import math
import cv2
import shapely.geometry as shgeo
import copy
from tqdm import tqdm
from pathlib import Path

from shapely.geometry import Polygon
from osgeo import ogr, gdal, gdalconst

"""
将 shp 文件中的 polygon 提取出来，转换为 TXT 的格式
"""
img_formats = tuple(['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng'])


def draw_box(input_dir, label_dir, output_dir, class_list, thickness=2, color=None):
    """
    对yolo和dota两种格式的数据进行画图
        0 0.8058167695999146 0.40044307708740234 0.6847715973854065 0.3581983745098114 91
        1367.196 1149.1193 1352.5274 134.8563 1893.1379 127.0375 1907.8070 1141.3006 matou 0
    @param input_dir: 图像文件夹
    @param label_dir: 标签文件夹
    @param output_dir: 画好的图像保存路径
    @param class_list: yolo 用的是 class_id 所以需要指定其对应的类名
    @return:
    """
    colors = color if color else [[random.randint(0, 255) for _ in range(3)] for _ in range(12)]
    class_name2color = dict()
    color_index = 0
    make_dir(output_dir)
    for image_name in tqdm(os.listdir(input_dir)):
        poly_path = os.path.join(label_dir, osp.splitext(image_name)[0] + '.txt')
        if not os.path.exists(poly_path):
            _ = open(poly_path, 'w')
        img_path = os.path.join(input_dir, image_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img_h, img_w = img.shape[:2]

        with open(poly_path, encoding='utf-8') as file_list:
            for line in file_list.readlines():
                data = line.strip().split("\n")[0].split(",")
                if len(data) == 5:
                    x, y, w, h = float(data[1]) * img_w, float(data[2]) * img_h, float(data[3]) * img_w, float(
                        data[4]) * img_h
                    poly = [[x - 0.5 * w, y - 0.5 * h], [x + 0.5 * w, y - 0.5 * h],
                            [x + 0.5 * w, y + 0.5 * h], [x - 0.5 * w, y + 0.5 * h]]
                    poly = np.array(poly, dtype=np.int0)
                    class_name = class_list[int(data[0])]
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=colors[int(data[0])],
                                     thickness=thickness)
                    c1 = (int(x), int(y))
                elif len(data) == 6:
                    x, y, w, h, theta = float(data[1]) * img_w, float(data[2]) * img_h, float(data[3]) * img_w, \
                                        float(data[4]) * img_h, int(data[5])
                    rect = ((x, y), (w, h), theta)
                    poly = np.float32(cv2.boxPoints(rect))  # 返回 rect 对应的四个点的值
                    poly = np.int0(poly)
                    class_name = class_list[int(data[0])]
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=colors[int(data[0])],
                                     thickness=thickness)
                    c1 = (int(rect[0][0]), int(rect[0][1]))

                elif len(data) == 9:
                    poly = data[0:8]
                    class_name = data[-1]
                    # if class_name not in class_name2color:
                    #     class_name2color[class_name] = color_index
                    #     color_index += 1  # 增加 color 标签

                    poly = list(map(float, poly))
                    poly = np.int0(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=(0, 0, 255),
                                     thickness=thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                elif len(data) == 10:
                    poly = data[0:-2]
                    class_name = data[-2]
                    if class_name not in class_name2color:
                        class_name2color[class_name] = color_index
                        color_index += 1  # 增加 color 标签

                    poly = list(map(float, poly))
                    poly = np.int0(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=colors[class_name2color[class_name]],
                                     thickness=thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                elif len(data) == 11:
                    class_name = data[-1]
                    if class_name not in class_name2color:
                        class_name2color[class_name] = color_index
                        color_index += 1  # 增加 color 标签
                    conf = data[1]
                    poly = [float(num) for num in data[2:-1]]
                    poly = np.int0(poly).reshape((4, 1, 2))
                    cv2.drawContours(image=img, contours=[poly], contourIdx=-1,
                                     color=colors[class_name2color[class_name]],
                                     thickness=thickness)
                    c1 = np.sum(poly, axis=0)[0] / 4  # 计算中心点坐标
                    c1 = int(c1[0]), int(c1[1])
                else:
                    print("not support this data format")
                    break

                # if class_name not in class_name2color:
                #     class_name2color[class_name] = color_index
                #     color_index += 1  # 增加 color 标签

                label = '%s' % (class_name)
                t_size = cv2.getTextSize(label, 0, fontScale=1 / 4, thickness=thickness)[0]
                try:
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 1 / 4, [225, 255, 255], thickness=thickness,
                                lineType=cv2.LINE_AA)
                except:
                    cv2.putText(img, 'other', (c1[0], c1[1] - 2), 0, 1 / 4, [225, 255, 255], thickness=thickness,
                                lineType=cv2.LINE_AA)

        img_name = Path(img_path).name
        cv2.imencode('.' + img_name.split('.')[-1], img)[1].tofile(os.path.join(output_dir, img_name))


def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (
                x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print(
            'θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
            x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
            x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside


def dota2yolo_obb(dir, class_list=None):
    """
    :param dir:
    :param class_list:
    :return:
    """
    # 分割之后的数据改成 yolo 训练的数据类型
    txt_dir = os.path.join(dir, 'labelTxt')
    img_dir = os.path.join(dir, 'images')
    img_list = os.listdir(img_dir)
    new_dir = os.path.join(dir, 'labels')
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.mkdir(new_dir)
    for file in tqdm(os.listdir(txt_dir)):
        txt_path = os.path.join(txt_dir, file)
        des_path = os.path.join(new_dir, file)
        h, w = 0, 0
        for img_name in img_list:
            if Path(img_name).stem == Path(txt_path).stem:
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                h_, w_ = img.shape[:2]
                break
        with open(des_path, 'w') as f_w:
            with open(txt_path) as f_r:
                for line in f_r.readlines():
                    data = line.split(' ')
                    # 只处理在 class_list 中的类别，这样即使数据不干净，也能得到干净的结果
                    if data[-2] not in class_list:
                        continue
                    class_id = class_list.index(data[-2])
                    # todo 调整 subsize
                    # [135.0, 203.0, 135.0, 142.0, 197.0, 142.0, 197.0, 203.0]
                    poly = [float(i) for i in data[0:-2]]
                    poly[::2] = [num / w_ for num in poly[::2]]
                    poly[1::2] = [num / h_ for num in poly[1::2]]
                    # print(class_id, poly)
                    poly = np.float32(poly).reshape((4, 2))
                    bo = np.sum(poly > 1)  # 输出归一化后仍然大于 1 的错误样本
                    if bo:
                        print(txt_path, w, h)
                    rect = cv2.minAreaRect(poly)
                    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[-1]
                    x, y, longside, shortside, theta_longside = cvminAreaRect2longsideformat(x, y, w, h, theta)
                    # [-180, 0) ==> [0, 180] ==> [0, 179]
                    theta_label = int(theta_longside + 180.5)
                    if theta_label == 180:
                        theta_label = 179
                    label_line = str(class_id) + " " + str(x) + " " + str(y) + " " + str(longside) + " " + str(
                        shortside) + " " + str(theta_label)
                    f_w.write(label_line + '\n')


def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])


def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r', encoding='utf-8')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r', encoding='utf-8')
        f = fd
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(',')
            object_struct = {}
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                object_struct['difficult'] = splitlines[9]
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            objects.append(object_struct)
        else:
            break
    return objects


def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
               poly[1][0], poly[1][1],
               poly[2][0], poly[2][1],
               poly[3][0], poly[3][1]
               ]
    return outpoly


def parse_dota_poly2(filename):
    """
        parse the dota ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects


def GetFileFromThisRootDir(dir, ext=None):
    """
    :param ext: 指定后缀文件的 list，默认 None 的时候就获取所有文件
    :return:
    """
    allfiles = []
    needExtFilter = (ext != None)
    for root, dirs, files in os.walk(dir):
        for filespath in files:
            filepath = os.path.join(root, filespath)
            extension = os.path.splitext(filepath)[1][1:]  # 不包含 "." 的后缀
            if needExtFilter and extension in ext:
                allfiles.append(filepath)
            elif not needExtFilter:
                allfiles.append(filepath)
    return allfiles


def choose_best_pointorder_fit_another(poly1, poly2):
    """
    旋转四个点的位置，找一个最匹配的组合
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate) ** 2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]


def cal_line_length(point1, point2):
    """
    计算两点间距离：(x1, y1), (x2, y2) ==> sqrt((x1-x2)^2 + (y1+y2)^2)
    """
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class CropBase:
    def __init__(
            self,
            imagepath,
            labelpath,
            save_dir,
            code='utf-8',
            gap=100,
            img_size=1024,
            thresh=1.0,
            choosebestpoint=True,
            ext='.png',
            save_none=False,
            split_minority=False,
            minority_list=None,
            fix_boundary=True
    ):
        """
        :param basepath: base path for dota data。确保在 basepath 下有 'images' 和 'labelTxt' 两个文件夹
        :param save_dir: output base path for dota data,
        :param code: 文本的编码格式
        :param gap: 重叠区域
        :param img_size: 小图的尺寸
        :param thresh: 这个阈值决定当裁剪的区域不完整时是否保存该元素的 label
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param save_none: 保存所有的样本，包括标签空白的样本
        :param split_minority:
        :param fix_boundary: 框可能会超出边界，是否补全为完整的矩形，还是将其移动到图像边界内
        """
        self.outpath = save_dir
        self.code = code
        self.gap = gap
        self.subsize = img_size
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.outimagepath = os.path.join(self.outpath, 'imgs')
        self.outlabelpath = os.path.join(self.outpath, 'annotations')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.save_none = save_none
        self.minority_list = minority_list
        self.fix_boundary = fix_boundary
        self.split_minority = split_minority

        if not os.path.exists(self.outimagepath):
            os.makedirs(self.outimagepath)
        if not os.path.exists(self.outlabelpath):
            os.makedirs(self.outlabelpath)

    def polyorig2sub(self, left, up, poly):
        """
        将大图中的 poly 转到小图的坐标，即以左上角坐标为原点
        """
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly) / 2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            计算的是针对交集与 poly1 的比值（inter_area / poly1_area）
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        """
        根据左上角坐标在大图中保存小图
        @param img: numpy.array()
        """
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imencode(self.ext, subimg)[1].tofile(outdir)

    def GetPoly4FromPoly5(self, poly):
        """
        将5个点的poly转成4个点的标准poly
        """
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1]), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i
                     in range(int(len(poly) / 2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2) % 10]) / 2)
                outpoly.append((poly[(count * 2 + 1) % 10] + poly[(count * 2 + 3) % 10]) / 2)
                count = count + 1
            elif (count == (pos + 1) % 5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        """
        @param split_hard_sample: 将小样本分割出来
        @param resizeimg: splitdata(rate)==》缩放，默认为 1 不缩放
        @param objects: util.parse_dota_poly2(fullname)
        """
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            save_img_txt = is_minority = False
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                        (obj['poly'][2], obj['poly'][3]),
                                        (obj['poly'][4], obj['poly'][5]),
                                        (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ','.join(list(map(str, polyInsub)))
                    outline = outline + ',' + obj['name']
                    if self.split_minority and obj['name'] in self.minority_list:
                        is_minority = True
                    f_out.write(outline + '\n')
                    save_img_txt = True  # 如果在文本中写入了信息就保存 txt 和图片

        if self.split_minority:
            if is_minority:
                self.saveimagepatches(resizeimg, subimgname, left, up)
            else:
                os.remove(outdir)
        else:
            # 如果文本没有写入，那么就不保存图像并删除文本，加快切图速度
            if self.save_none or save_img_txt:
                self.saveimagepatches(resizeimg, subimgname, left, up)
            # 随机保留一些负样本（空样本）
            else:
                os.remove(outdir)

    def SplitSingle(self, name, rate, extent):
        """
        将 img 和对应的 label 进行切割
        """
        img = cv2.imdecode(np.fromfile(os.path.join(self.imagepath, name + extent), dtype=np.uint8), -1)
        if np.shape(img) == (): return
        fullname = os.path.join(self.labelpath, name + '.txt')
        if not osp.exists(fullname): return  # txt 不存在
        try:
            objects = parse_dota_poly2(fullname)
        except:
            print(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x: rate * x, obj['poly']))
            # obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '_' + str(rate) + '_'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '_' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def crop_data(self, rate, class_list, isObb):
        """
        rate: 先缩放 后裁剪
        缩放后开始活动裁剪，并转换数据格式
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [custombasename(x) for x in imagelist if (custombasename(x) != 'Thumbs')]
        for name in tqdm(imagenames):
            for img_name in imagelist:
                if name == Path(img_name).stem:
                    self.ext = '.' + img_name.split('.')[-1]
            self.SplitSingle(name, rate, self.ext)
        # print("切图结束，开始转换数据格式。。。")
        # if isObb:
        #     dota2yolo_obb(dir=self.outpath, class_list=class_list)
        # else:
        #     dota2yolo_hbb(dir=self.outpath, class_list=class_list)
        # print("开始画图。。。")
        draw_box(input_dir=os.path.join(self.outpath, 'imgs'),
                 label_dir=os.path.join(self.outpath, 'annotations'),
                 output_dir=os.path.join(self.outpath, 'draw'),
                 class_list=class_list)  # yolo 格式需要指定


def make_dir(dir_or_list, exist_ok=False):
    """
    创建文件夹或是多个文件夹，当其存在时会删除重新创建，避免冲突
    @param dir_or_list: str(path) or a list of path
    @:param exist_ok: 如果该文件夹存在的情况下，如果设置为 False，则删除后重新创建，设置为 True 则不管
    """

    def make_dir_(dir, exist_ok):
        if os.path.exists(dir) and not exist_ok:
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=exist_ok)

    if type(dir_or_list) == list:
        for i in range(len(dir_or_list)):
            make_dir_(dir_or_list[i], exist_ok)
    elif type(dir_or_list) == str:
        make_dir_(dir_or_list, exist_ok)


def geo2imgxy(trans, x, y):
    # 地理坐标转换为像素坐标
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 解多元一次方程：a*x=b


def getrectobj(img_path, shp_path):
    """
    获取目标的坐标，然后计算包裹它的最小水平框
    @param img_path: 栅格图像路径
    @param shp_path: shp 矢量的文件路径
    """
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")  # 消除字段乱码
    # gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 解析中文编码

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp_path, 0)
    dataset = gdal.Open(img_path, gdalconst.GA_ReadOnly)
    geo_transform = dataset.GetGeoTransform()
    if geo_transform == (0, 1, 0, 0, 0, 1):
        geo_transform = (0, 1, 0, 0, 0, -1)

    layer = dataSource.GetLayerByIndex(0)
    boxes = []
    for feature in layer:
        geo = feature.geometry()  # 获得面
        fid = feature.GetField('FldName')
        for p in geo:  # 获得line
            pnum = p.GetPointCount()  # 获得line里的点
            if (pnum >= 5):  # 5 个点才能组成一个圈
                p1x, p1y = geo2imgxy(geo_transform, p.GetX(0), p.GetY(0))
                p2x, p2y = geo2imgxy(geo_transform, p.GetX(1), p.GetY(1))
                p3x, p3y = geo2imgxy(geo_transform, p.GetX(2), p.GetY(2))
                p4x, p4y = geo2imgxy(geo_transform, p.GetX(3), p.GetY(3))
                lines = f'{round(p1x)},{round(p1y)},{round(p2x)},{round(p2y)},{round(p3x)},{round(p3y)},{round(p4x)},{round(p4y)},"{fid}"\n'
                boxes.append(lines)
    return boxes


def shp2dota_batch(img_dir, shp_dir, save_dir):
    make_dir(save_dir, exist_ok=True)
    for img_file in tqdm(os.listdir(img_dir)):
        if img_file.lower().endswith(img_formats):
            img_path = os.path.join(img_dir, img_file)
            img_name, _ = os.path.splitext(img_file)
            shp_path = os.path.join(shp_dir, img_name + '.shp')
            txt_path = os.path.join(save_dir, img_name + '.txt')
            if osp.exists(shp_path):  # shp 文件是否存在
                txt_file = open(txt_path, 'w', encoding='utf-8')
                if os.path.exists(shp_path):
                    boxes = getrectobj(img_path, shp_path)
                    txt_file.writelines(boxes)
    return os.listdir(img_dir)[0].split('.')[-1]


def main(file_name, img_dir, shp_dir):
    img_dir = Path(img_dir)
    big_txt_dir = osp.join(img_dir.parent, 'annotations')
    ext = shp2dota_batch(
        img_dir=img_dir,
        shp_dir=shp_dir,
        save_dir=big_txt_dir,  # txt保存路径
    )
    crop = CropBase(
        imagepath=img_dir,
        labelpath=big_txt_dir,
        save_dir=osp.join(img_dir.parent, 'data_processing'),
        img_size=640,
        gap=60,
        ext='.' + ext,
        save_none=False,
        split_minority=False,
        minority_list=None,
        fix_boundary=False
    )
    crop.crop_data(1, class_list=[], isObb=True)


if __name__ == "__main__":
    # img_dir = r'E:\datasets\ocr\OCR_dataset_test\test\imgs'  # 栅格图像路径
    # shp_dir = r'E:\datasets\ocr\OCR_dataset_test\test\shps'  # 矢量的文件路径
    # main(12, img_dir, shp_dir)

    # print(sys.argv)
    main(*sys.argv)

'''
python crop_img_label_ocr.py E:\datasets\ocr\OCR_dataset_test\test\imgs E:\datasets\ocr\OCR_dataset_test\test\shps
'''
