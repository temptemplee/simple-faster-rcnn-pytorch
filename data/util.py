import numpy as np
from PIL import Image
import random


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        读入的path其实是一个个img文件, os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        VOCdevkit\VOC2007\JPEGImages\000001.jpg - 009963.jpg约1万张图片,图像是CHW格式的
        在神经网络中，图像被表示成[c, h, w]格式或者[n, c, h, w]格式，
        但如果想要将图像以np.ndarray形式输入，因np.ndarray默认将图像表示成[h, w, c]个格式，
        需要对其进行转化。n：样本数量 c：图像通道数 w：图像宽度 h：图像高度
        from PIL import Image
        imoprt numpy as np

        img_path = ('./test.jpg')
        img = Image.open(img_path)
        img_arr = np.array(img)
        print(img_arr.shape)

        # 输出的结果是(500, 300, 3)
        从上面的试验结果我们可以知道，图像以[h, w, c]的格式存储在np.ndarray中的。
        参见https://blog.csdn.net/baidu_26646129/article/details/86712889

        NCHW，又称：“channels_first”，是nvidia cudnn库原生支持的数据模式；
        在GPU中，使用NCHW格式计算卷积，比NHWC要快2.5倍左右（0:54 vs 2:14）
        NHWC, 又称“channels_last”，是CPU指令比较适合的方式，SSE 或 AVX优化，沿着最后一维，即C维计算，会更快。
        NCHW排列，C在外层，所以每个通道内，像素紧挨在一起，即“RRRGGGBBB”；
        NHWC排列，C在最内层，所以每个通道内，像素间隔挨在一起，即“RGBRGBRGB”，如下所示：
        https://www.jianshu.com/p/61de601bc90f 

        对于"NCHW" 而言，其同一个通道的像素值连续排布，更适合那些需要对每个通道单独做运算的操作，比如"MaxPooling"。
        对于"NHWC"而言，其不同通道中的同一位置元素顺序存储，因此更适合那些需要对不同通道的同一像素做某种运算的操作，比如“Conv1x1

        由于NCHW，需要把所有通道的数据都读取到，才能运算，所以在计算时需要的存储更多。这个特性适合GPU运算，正好利用了GPU内存带宽较大并且并行性强的特点，其访存与计算的控制逻辑相对简单；
        而NHWC，每读取三个像素，都能获得一个彩色像素的值，即可对该彩色像素进行计算，这更适合多核CPU运算，CPU的内存带宽相对较小，每个像素计算的时延较低，临时空间也很小；
        若采取异步方式边读边算来减小访存时间，计算控制会比较复杂，这也比较适合CPU。
        结论：在训练模型时，使用GPU，适合NCHW格式；在CPU中做推理时，适合NHWC格式。
        采用什么格式排列，由计算硬件的特点决定。
        OpenCV在设计时是在CPU上运算的，所以默认HWC格式。
        TensorFlow的默认格式是NHWC，也支持cuDNN的NCHW
        
        dtype: The type of array. The default value is :obj:`~numpy.float32`. 此类形是单精度浮点数
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.
            如果color = false，下面的代码实际上不是灰度值，而是'P'模式，
            8位彩色图像(也叫做单通道格式彩图)，它的每个像素用8个bit表示
            详见 https://blog.csdn.net/Leon1997726/article/details/109016170 
            或者 https://zhuanlan.zhihu.com/p/58012264 

    Returns:
        ~numpy.ndarray: An image.
    """
        # np.array()和np.asarray()的区别：
        # 都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，
        # array仍然会copy出一个副本，占用新的内存，但asarray不会。
        # >>> data1=[[1,1,1],[1,1,1],[1,1,1]]
        # >>> type(data1)
        # <class 'list'> 这里data1是非ndarray型的，所以如下的arr2和arr3都是深拷贝
        # >>> arr2=np.array(data1)
        # >>> arr3=np.asarray(data1)
        # >>> data1[1][1]=2
        # >>> print(data1)
        # [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        # >>> print(arr2)
        # [[1 1 1]
        # [1 1 1]
        # [1 1 1]]
        # >>> print(arr3)
        # [[1 1 1]
        # [1 1 1]
        # [1 1 1]]
        # >>>
        # >>> data2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
        # >>> type(data2)
        # <class 'numpy.ndarray'> 这里data2是ndarray型的，所以arr4还是深拷贝，但arr5是浅拷贝一同随data2变化
        # >>> arr4=np.array(data2)
        # >>> arr5=np.asarray(data2)
        # >>> data2[1][1]=3
        # >>> print(data2)
        # [[1 1 1]
        # [1 3 1]
        # [1 1 1]]
        # >>> print(arr4)
        # [[1 1 1]
        # [1 1 1]
        # [1 1 1]]
        # >>> print(arr5)
        # [[1 1 1]
        # [1 3 1]
        # [1 1 1]]


    # try：
    # code    #需要判断是否会抛出异常的代码，如果没有异常处理，python会直接停止执行程序
    
    # except:  #这里会捕捉到上面代码中的异常，并根据异常抛出异常处理信息
    # #except ExceptionName，args：    #同时也可以接受异常名称和参数，针对不同形式的异常做处理
    
    # code  #这里执行异常处理的相关代码，打印输出等
    
    # else：  #如果没有异常则执行else
    
    # code  #try部分被正常执行后执行的代码
    
    # finally：
    # code  #退出try语句块总会执行的程序

    # 1.这里的else是和trycatch连用的，并且else只在try中代码没有异常的情况下执行，else必须在except这句代码存在的时候才能出现。

    # 2.finally这个片段里面的代码是肯定在最后执行的，无论前面是否发生异常，最后总会执行finally片段内的代码。
    # 如果try里面有return的话，那么finally的code也会在return之前执行到。详见
    # https://www.cnblogs.com/xuanmanstein/p/8080629.html

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P') # P是单通道格式彩图
        # 运行到这里：type(img)-> <class 'PIL.Image.Image'> print(img) <PIL.Image.Image image mode=RGB size=353x500 at 0x25414744D00>
        img = np.asarray(img, dtype=dtype) # ndarray化的img是[h,w,c]格式的
        # 运行到这里 type(img)-> <class 'numpy.ndarray'>; print(img.ndim)-> 3; print(img.shape) -> (500, 353, 3)
        # 第一维是行，对应图片的高；第二维是列，对应图片的宽，第三维是3个RGB通道。
        # >>> print(img)
        # [[[  1.   1.   0.] 最内层的对应的是某列某行的像素的点对应的[R G B]三个值
        # [  1.   1.   0.]
        # [  1.   1.   0.]
        # ...
        # [  2.   4.   3.]
        # [  2.   4.   3.]
        # [  1.   3.   2.]]
        # ...

    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W) #f.convert()在1、P、L模式下都是单通道的ndarray信息
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.

    This method is mainly used together with image cropping.
    This method translates the coordinates of bounding boxes like
    :func:`data.util.translate_bbox`. In addition,
    this function truncates the bounding boxes to fit within the cropped area.
    If a bounding box does not overlap with the cropped area,
    this bounding box will be removed.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_slice (slice): The slice of y axis.
        x_slice (slice): The slice of x axis.
        allow_outside_center (bool): If this argument is :obj:`False`,
            bounding boxes whose centers are outside of the cropped area
            are removed. The default value is :obj:`True`.
        return_param (bool): If :obj:`True`, this function returns
            indices of kept bounding boxes.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`, returns an array :obj:`bbox`.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`bbox, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **index** (*numpy.ndarray*): An array holding indices of used \
            bounding boxes.

    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox


def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img
