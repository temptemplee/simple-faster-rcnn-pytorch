import numpy as np
import torch
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox

# 上一级ProposalCreator产生2000个ROIS，但是这些ROIS并不都用于训练，
# 经过本级ProposalTargetCreator的筛选产生128个用于自身的训练，规则如下:、
# 1 ROIS和GroundTruth_bbox的IOU大于0.5(pos_iou_thresh),选取一些(比如说本实验的32(pos_ratio)个)作为正样本
# 2 选取ROIS和GroundTruth_bbox的IOUS小于等于0(neg_iou_thresh_lo)的选取一些比如说选取128-32=96个作为负样本
# 3 然后分别对ROI_Headers进行训练
class ProposalTargetCreator(object):
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_lo`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    # 假设一张960*540的图片，经过vgg16的四次pool，最后的特征图大小为60*33，每个像素点产生9个anchor，共60*33*9=17820个，约20000个anchor
    # 为2000个rois赋予ground truth！（严格讲挑出128个赋予ground truth！）
    # 输入：2000个rois、一个batch（一张图）中所有的bbox ground truth（R，4）、对应bbox所包含的label（R，1）（VOC2007来说20类0-19）
    # 输出：128个sample roi（128，4）、128个gt_roi_loc（128，4）、128个gt_roi_label（128，1）
    # 因为这些数据是要放入到整个大网络里进行训练的，比如说位置数据，所以要对其位置坐标进行数据增强处理(归一化处理)
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape

        # numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接
        # >>> a=np.array([[1,2,3],[4,5,6]])
        # >>> b=np.array([[11,21,31],[7,8,9]])
        # >>> np.concatenate((a,b),axis=0)
        # array([[ 1,  2,  3],
        #     [ 4,  5,  6],
        #     [11, 21, 31],
        #     [ 7,  8,  9]])
        # >>> np.concatenate((a,b),axis=1)  #axis=1表示对应行的数组进行拼接
        # array([[ 1,  2,  3, 11, 21, 31],
        #        [ 4,  5,  6,  7,  8,  9]])
        # numpy.concatenate()比numpy.append()效率高，适合大规模的数据拼接
        # 首先将2000个roi和m个bbox给concatenate了一下成为新的roi（2000+m，4）
        # TODO: 这里为啥concatenate?感觉没必要
        # 在https://www.jianshu.com/p/c04eaf1b3812 这篇文章中提及这个问题了 
        # 说是在训练初期，几乎就是随机输出，可能连一个正样本都没有，加入GT一方面
        # 弥补了正样本数量的不足，另一方面还提供了更优质的正样本
        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox) # iou (2000+m, m) # 最后m行m列其实都是bbox自己
        gt_assignment = iou.argmax(axis=1) # 按行找到最大值，返回最大值对应的序号。返回的是每个roi与**哪个**bbox的最大。shape是(2000+m,)
        max_iou = iou.max(axis=1) # 每个roi与对应bbox最大的iou，shape是(2000+m,)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        # gt_assignment是R个bbox里面的第几个，label[gt_assignment]是第gt_assignment个bbox所对应的类别值，再加1做偏移
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        # 利用这128个索引值keep_index就得到了128个sample roi，128个gt_label
        keep_index = np.append(pos_index, neg_index) #TODO:干嘛不用np.concatenate了？
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        # 将sample_roi和其所属bbox经函数bbox2loc就得到了128个gt_loc
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])

        # 因为这些数据是要放入到整个大网络里进行训练的，比如说位置数据，所以要对其位置坐标进行数据增强处理(归一化处理)
        # bbox2loc没有进行归一化，只是按照公式计算tx,ty,th,tw，保证Pw和Ph为正。至于均值和方差的取值，和数据集有关。
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label

# 为Faster-RCNN专有的RPN网络提供自我训练的样本，RPN网络正是利用AnchorTargetCreator产生的样本
# 作为数据进行网络的训练和学习的，这样产生的预测anchor的类别和位置才更加精确，anchor变成真正的ROIS
# 需要进行位置修正，而AnchorTargetCreator产生的带标签的样本就是给RPN网络进行训练学习用哒
class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5): # 默认值正样本数不超过128个
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    # anchor的4个值的意思是: (y_{min}, x_{min}, y_{max}, x_{max})
    # bbox的4个值好像意思也一样
    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W) # 将那些超出图片范围的anchor全部去掉,只保留位于图片内部的序号
        anchor = anchor[inside_index] # 保留位于图片内部的anchor
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox) # 筛选出符合条件的正例128个负例128并给它们附上相应的label

        # compute bounding box regression targets
        # anchor和bbox[argmax_ious]两个都是inside_index长度的
        # >>> bbox
        # array([[ 1,  5,  5,  2],
        #     [ 9,  6,  2,  8],
        #     [ 3,  7,  9,  1],
        #     [30, 17,  9, 21]])
        # >>> bbox[0,2] 用简单的坐标得出来的是一个值，要用如下的ndarray或者list
        # 5
        # >>> a = np.array([0,2])
        # >>> a
        # array([0, 2])
        # >>> bbox[a] ndarray模式，可以返回bbox中指定的argmax_ious行
        # array([[1, 5, 5, 2],
        #     [3, 7, 9, 1]])
        # >>> a = (0,2)
        # >>> bbox[a]
        # 5
        # >>> a = [0,2]
        # >>> bbox[a] list模式，可以返回bbox中指定的argmax_ious行
        # array([[1, 5, 5, 2],
        #     [3, 7, 9, 1]])
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1) # 将位于图片内部的框的label对应到所有生成的20000个框中（label原本为所有在图片中的框的）
        loc = _unmap(loc, n_anchor, inside_index, fill=0) # 将回归的框对应到所有生成的20000个框中（label原本为所有在图片中的框的）

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        # numpy.empty(shape, dtype=float, order=‘C’) 根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。
        # empty不像zeros一样，并不会将数组的元素值设定为0，因此运行起来可能快一些。
        # 在另一方面，它要求用户人为地给数组中的每一个元素赋值，所以应该谨慎使用。
        # 在初始化一个不为0的数组时，np.empty然后np.fill比np.zero再np.fill要快一些。
        label = np.empty((len(inside_index),), dtype=np.int32) # inside_index为所有在图片范围内的anchor序号
        label.fill(-1) # 缺省填-1

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index) # 调用_calc_ious（）函数得到每个anchor与哪个bbox的iou最大以及这个iou值、
                                                        # 每个bbox与哪个anchor的iou最大(需要体会从行和列取最大值的区别)
                                                        # argmax_ious：每行最大ious的索引值(总共inside_index个，因为inside_index行)
                                                        # max_ious：上述每行对应argmax_ious位置的ious值
                                                        # gt_argmax_ious：返回所有gt_box(列)对应最大ious所在的anchor值(行)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # 根据这个看，_calc_ious()里面gt_argmax_ious只需要计算第一遍就足够这里标记1了
        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        # numpy.random.choice(a, size=None, replace=True, p=None)
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        # 从pos_index中随机取len(pos_index) - n_pos个数组成disable_index，然后给它设置成-1
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        # ious维度是(anchor个数N，bbox个数K) N大概有15000个传入的anchor
        # 实际上是anchor = anchor[inside_index]，所以其实N = len(inside_index)
        ious = bbox_iou(anchor, bbox)

        # >>> a
        # array([[1, 5, 5, 2],
        #        [9, 6, 2, 8],
        #        [3, 7, 9, 1]])
        # >>> print(np.argmax(a, axis=1))
        # [1 0 2]
        argmax_ious = ious.argmax(axis=1) # 1代表行，0代表列 axis=1表明argmax求的是每一个anchor对于所有gt_box最大的iou的索引值
                                          # len(argmax) = N(即len(inside_index))

        # arange() 主要是用于生成数组 numpy.arange(start, stop, step, dtype = None)
        # start —— 开始位置，数字，可选项，默认起始值为0
        # stop —— 停止位置，数字
        # step —— 步长，数字，可选项， 默认步长为1，如果指定了step，则还必须给出start。
        # ious是一个(N(即len(inside_index)), K)维度
        # np.arange(len(inside_index))生成了array([0, 1, 2,...N-1])，这里其实也可以用np.arange(ious.shape[0])代替
        # 然后ious[x, y]是分别返回坐标在(x,y)的ious数值，max_ious的shape是(N,1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]

        # >>> a
        # array([[1, 5, 5, 2],
        #        [9, 6, 2, 8],
        #        [3, 7, 9, 1]])
        # >>> print(np.argmax(a, axis=0))
        # [1, 2, 2, 1]，返回的是列下每列最大值(每个gt_box对不同anchor的最大值的索引)
        gt_argmax_ious = ious.argmax(axis=0) # 0代表列，从列取最大

        # gt_max_ious = array([9, 7, 9, 8]) 取得列模式下最大值坐标所对应的最大值本身
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] #求出每个bbox与哪个anchor的iou最大值，gt_max_ious的shape是(K,1)

        # ious == gt_max_ious 这里用了广播的特性
        # array([[False, False, False, False],
        #        [ True, False, False,  True],
        #        [False,  True,  True, False]])
        # np.where(ious == gt_max_ious)
        # (array([1, 1, 2, 2], dtype=int64), array([0, 3, 1, 2], dtype=int64))
        # np.where返回的是condition为true的N维坐标，这里是2维的，所以[1,0],[1,3],[2,1],[2,2]，
        # 但是np.where()[0]是第一维的，而且数值是从小到大排列的。
        #TODO: 这样做有个问题：如果 ious一列中有两个相同的最大值，如第三列：
        # >>> ious
        # array([[1, 5, 5, 2],
        #        [9, 6, 9, 8],
        #        [3, 7, 9, 1]])
        # gt_max_ious = array([9, 7, 9, 8])
        # >>> ious == gt_max_ious 第三列就会有多个True值
        # array([[False, False, False, False],
        #        [ True, False,  True,  True],
        #        [False,  True,  True, False]])
        # >>> np.where(ious == gt_max_ious)
        # (array([1, 1, 1, 2, 2], dtype=int64), array([0, 2, 3, 1, 2], dtype=int64))
        # 第一维就会返回多于K个的值，后期如何处理？
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1: # label是一维数据
        # np.empty随机生成size=count的数组
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else: # loc是多维数组 :math:`(R, 4)`，且用[1:]开始跳过第0维，第0维用count代替
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        roi = loc2bbox(anchor, loc) # 利用预测的修正值，对12000个anchor进行修正

        # Clip predicted boxes to image.
        # roi维度是(R,4), slice(start, end, step), slice(0, 4, 2)取出第0列和第2列，
        # 然后用numpy.clip(a, a_min, a_max, out=None)截取a_min, a_max之间的
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        # slice(1, 4, 2)取出第1列和第3列
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep] # TODO: 传入的score是啥

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        # ravel()将数组维度拉成一维数组，argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # 例如 >>> np.array([1,4,3,-1,6,9]).argsort()
        # array([3, 0, 2, 1, 4, 5], dtype=int64) 输出的是索引值
        # [::-1]逆序全部排列，逆序后就是从大到小排列了
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        # # NMS算法
        # bboxes维度为[N,4]，scores维度为[N,], 均为tensor
        # def nms(self, bboxes, scores, threshold=0.5):
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]、
        # pytorch用GPU训练数据时，需要将数据转换成tensor类型(上面的torch.from_numpy().cuda()过程)，
        # 其输出keep也是tensor类型。如果想把CUDA tensor格式的数据改成numpy时，
        # 需要先将其转换成cpu float-tensor随后再转到numpy格式。 
        # numpy不能读取CUDA tensor 需要将它转化为 CPU tensor 所以得写成.cpu().numpy()
        roi = roi[keep.cpu().numpy()]
        return roi
