import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """
    # __init__构造方法用于创建对象时使用，每当创建一个类的实例对象时，Python 解释器都会自动调用它。
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):

        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        # id_list_file为trainval.txt，或者test.txt 
        # TODO: trainval.txt test.txt都是些啥东西？
        # 格式化字符串的函数 str.format()，它增强了字符串格式化的功能。
        # "{} {}".format("hello", "world")    # 不设置指定位置，按默认顺序 'hello world'
        # "{0} {1}".format("hello", "world")  # 设置指定位置 'hello world'
        # "{1} {0} {1}".format("hello", "world")  # 设置指定位置 'world hello world'
        # VOC2007包含{'train', 'val', 'trainval', 'test'}，共20类，加背景21类。
        # 四个集合图片数分别为2501， 2510，5011，4952（trainval=train+val）。
        # VOC2012无test集。

        # strip() 方法删除任何前导（开头的空格）和尾随（结尾的空格）字符（空格是要删除的默认前导字符）
        self.ids = [id_.strip() for id_ in open(id_list_file)
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES # 共20类

    # 如果一个类表现得像一个list，要获取有多少个元素，就得用 len() 函数。
    # 要让 len() 函数工作正常，类必须提供一个特殊方法__len__()，它返回元素的个数
    def __len__(self):
        return len(self.ids) #trainval.txt有5011个，test.txt有210个

    def get_example(self, i): # 返回彩色图像和边界框。
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml')) #读取.xml文件（标签）
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it. 对xml标签文件进行解析，xml文件中包含object name和difficult(0或者1,0代表容易检测)
            if not self.use_difficult and int(obj.find('difficult').text) == 1:  #标为difficult的目标在测试评估中一般会被忽略
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            # TODO: 标注的bbox坐标都是实际坐标值？不是以0开始的？
            # TODO: bbox的4个坐标顺序 从utils.py看来顺序是[ymin, xmin, ymax, xmax],再查
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip() # text转小写再去头尾
            label.append(VOC_BBOX_LABEL_NAMES.index(name)) # 20个分类里面都是小写
        bbox = np.stack(bbox).astype(np.float32) # 所有object的bbox坐标存在列表里
        label = np.stack(label).astype(np.int32) # 所有object的label存在列表里
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True) #如果color=True，则转换为RGB图

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    # 凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像
    # 这样p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法__getitem__。
    # 一般如果想使用索引访问元素时，就可以在类中定义这个方法（__getitem__(self, key) ）
    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
