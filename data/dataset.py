from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

# 实现对caffe与torchvision版本的去正则化。因为可以利用caffe版本的vgg预训练权重，也可利用torchvision版本的预训练权重。只不过后者结果略微逊色于前者。
# 因为pytorch预训练模型采用RGB 0-1图片(0-255?)，而对于caffe预训练模型输入为BGR 0-255图片
# 不要偷懒，尽可能的“Match Everything”。由于torchvision中有预训练好的VGG16，
# 而caffe预训练VGG要求输入图片像素在0-255之间（torchvision是0-1），BGR格式的，标准化只减均值，不除以标准差，
# 看起来有点别扭（总之就是要多写几十行代码+专门下载模型）。然后我就用torchvision的预训练模型初始化，
# 最后用了一大堆的trick，各种手动调参，才把mAP调到0.7（正常跑，不调参的话大概在0.692附近）。
# 某天晚上抱着试试的心态，睡前把VGG的模型改成caffe的，第二天早上起来一看轻轻松松0.705 ...
def inverse_normalize(img):
    if opt.caffe_pretrain:
        # caffe_normalize之前有减均值预处理，现在还原回去
        # 为啥是如下这三个数？
        # https://github.com/open-mmlab/mmdetection/issues/4613 这三个数是caffe 基于骨架detectron统计出来的(还有另一组不同的值)。
        # 这里没有乘以标准差，可见如上连接，std为1
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :] #将caffe的BGR转换为pytorch需要的RGB图片（python [::-1]为逆序输出）
    # approximate un-normalize for visualize
    # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
    # pytorch_normalze中标准化采用0均值标准化，转化函数为（x-mean）/(standard deviation)，现在乘以标准差加上均值还原回去，转换为0-255
    # TODO: 0.225和0.45的由来？
    # 参考下面pytorch_normalze()的注解
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


# 采用pytorch预训练模型对图片预处理，函数输入的img为0-1，参加下面的preprocess()
# 实现对pytorch模型输入图像的标准化：由【0，255】的RGB转为【0，1】的RGB再正则化为【-1，1】的RGB
# PyTorch 中我们经常看到 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ，是从 ImageNet 数据集的数百万张图片中随机抽样计算得到的
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]) # transforms.Normalize使用如下公式进行归一化channel=（channel-mean）/std,转换为[-1,1]
    # torch.from_numpy(ndarray) → Tensor，即 从numpy.ndarray创建一个张量。
    # 说明：返回的张量和ndarray共享同一内存。对张量的修改将反映在ndarray中，反之亦然。返回的张量是不能调整大小的。
    img = normalize(t.from_numpy(img)) # (ndarray) → Tensor
    return img.numpy()


# 采用caffe预训练模型对图片预处理，函数输入的img为0-1，参加下面的preprocess()
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    # TODO: 这里为啥不除以std?
    # 可参见 https://github.com/open-mmlab/mmdetection/issues/4613 
    # mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False) 的std均为1！
    img = (img - mean).astype(np.float32, copy=True)
    return img

# 函数输入的img为0-255
def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2) # 选小的比例，这样长和宽都能放缩到规定的尺寸 注意长宽缩放比是一样的，没有两个不同的
    img = img / 255. # 转换为0-1
    # resize到（H * scale, W * scale）大小，anti_aliasing为是否采用高斯滤波
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    # Transform实现了预处理，定义了__call__方法，
    # Python 类中一个非常特殊的实例方法，即 __call__()。
    # 该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用
    # 在__call__方法中利用函数preprocess对图像预处理，并将bbox按照图像缩放的尺度等比例缩放。
    # 然后随机对图像与bbox同时进行水平翻转
    # TODO: in_data.label不做任何处理就直接返回了？要它何用
    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H # 得出缩放比因子，注意长宽共用preprocess()同一个缩放比
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W)) # 调整框的大小，按照与原框等比例缩放

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True) # 只随机水平翻转
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir) # 会自动调用VOCBboxDataset::__init__()，实例化类
        self.tsf = Transform(opt.min_size, opt.max_size) # 这里调用了Transform::__init__()，实例化类

    def __getitem__(self, idx):
        # VOCBboxDataset 既然也定义了__getitem__(),这里就应该调用__getitem__(), 
        # 而不是get_example()了 TODO: ?
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        # 这里通过重载()调用了Transform::__call__（），用(ori_img, bbox, label)封装成传入的参数
        # 为了以示区别，传入的用ori_img，tranform处理后返回的用img。
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


# TestData完成的功能和前面类似，但是获取调用的数据集是不同的 
# 从Voc_data_dir中获取数据的时候使用了split='test'也就是从test往后分割的部分数据送入到TestDataset的self.db中
# 在进行图片处理的时候，并没有调用transform函数，因为测试图片集没有bboxes需要考虑，
# 同时测试图片集也不需要随机反转，反转无疑为测试准确率设置了阻碍！所以直接调用preposses()函数
# 进行最大值最小值裁剪然后归一化就完成了测试数据集的处理！
# 还有use_difficult默认为True TODO:为啥?
class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
