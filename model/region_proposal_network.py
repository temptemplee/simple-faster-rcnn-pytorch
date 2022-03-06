import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16, # feat_stride=16，因为经过四次pooling所以feature map的尺寸是原图的1/16
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True) 
        # 特征（N，512，h，w）输入进来（原图像的大小：16*h，16*w），首先是加pad的512个3*3大小卷积核，输出仍为（N，512，h，w）。
        # 然后左右两边各有一个1*1卷积。左路为18个1*1卷积，输出为（N，18，h，w），即所有anchor的0-1类别概率（h*w约为2400，
        # h*w*9约为20000）。右路为36个1*1卷积，输出为（N，36，h，w）
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape # x为feature map，n为batch_size,此版本代码为1. hh，ww即为宽高
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww) anchor的数量是hh*ww
        h = F.relu(self.conv1(x)) # 512个3x3卷积(512, H/16,W/16),后面都不写batch_size了

        rpn_locs = self.loc(h) # n_anchor（9）*4个1x1卷积，回归坐标偏移量。（9*4，hh,ww）
        # UNNOTE: check whether need contiguous
        # A: Yes
        # permute(dims) 将tensor的维度换位
        #  当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) # 转换为（n，hh，ww，9*4）后变为（n，hh*ww*9，4）
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() # 转换为（n,hh,ww,9*2）

        # 计算{Softmax}(x_{i}) = \{exp(x_i)}{\sum_j exp(x_j)}
        # torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
        # input是我们输入的数据，dim是在哪个维度进行Softmax操作（如果没有指定，默认dim=1）
        # 如下是对第4维，这里也即2做softmax，其实也就是二分法
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4) # 计算softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous() # 得到前景的分类概率
        rpn_fg_scores = rpn_fg_scores.view(n, -1) # 得到所有anchor的前景分类概率
        rpn_scores = rpn_scores.view(n, -1, 2) # 得到每一张feature map上所有anchor的网络输出值

        rois = list()
        roi_indices = list()
        for i in range(n): # n=1,代表batch_size数
            # 调用ProposalCreator函数， rpn_locs维度（hh*ww*9，4），rpn_fg_scores维度为（hh*ww*9），
            # anchor的维度为（hh*ww*9，4）， img_size的维度为（3，H，W），H和W是经过数据预处理后的。
            # 计算（H/16）x(W/16)x9(大概20000)个anchor属于前景的概率，取前12000个并
            # 经过NMS得到2000个近似目标框G^的坐标。roi的维度为(2000,4)
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0) # axis=0表示在列的维度上进行拼接
        # 这个 roi_indices在此代码中是多余的，因为我们实现的是batch_siae=1的网络，
        # 一个batch只会输入一张图象。如果多张图象的话就需要存储索引以找到对应图像的roi
        roi_indices = np.concatenate(roi_indices, axis=0)
        # rpn_locs的维度（hh*ww*9，4），rpn_scores维度为（hh*ww*9，2）， 
        # rois的维度为（2000,4），roi_indices用不到，anchor的维度为（hh*ww*9，4）
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 利用base anchor生成所有对应feature map的anchor
    # anchor_base :(9,4) 坐标，这里 A=9
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride) # 就是以feat_stride为间距产生从(0,height*feat_stride)的一行，纵向偏移量（0，16，32，...）
    shift_x = xp.arange(0, width * feat_stride, feat_stride) # 就是以feat_stride产生从(0,width*feat_stride)的一行，横向偏移量（0，16，32，...）
    # numpy.meshgrid()——生成网格点坐标矩阵 shift_x = [[0，16，32，..],[0，16，32，..],[0，16，32，..]...],
    # shift_x = [[0，0，0，..],[16，16，16，..],[32，32，32，..]...],就是形成了一个纵横向偏移量的矩阵，
    # 也就是特征图的每一点都能够通过这个矩阵找到映射在原图中的具体位置
    # 代码就是shift_x是以shift_x为行，以shift_y的行为列产生矩阵，同样shift_y是以shift_y的行为列，以shift_x的行的个数为列数产生矩阵
    # 例如：x=[-3, -2, -1]; y=[-2, -1]; X,Y=np.meshgrid(x,y), 则得到X=（[[-3, -2, -1],[-3, -2, -1]]）;
    # Y=([[-2, -1],[-2, -1],[-2, -1]])  产生的X以x的行为行，以y的元素个数为列构成矩阵，
    # 同样的产生的Y以y的行作为列，以x的元素个数作为列数产生矩阵
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)

    # 产生偏移坐标对，一个朝x方向，一个朝y方向偏移。此时X,Y的元素个数及矩阵大小都是相同的，
    # （X.ravel()之后变成一行，此时shift_x,shift_y的元素个数是相同的，都等于特征图的长宽的乘积(像素点个数)，
    # 不同的是此时的shift_x里面装得是横向看的x的一行一行的偏移坐标，而此时的y里面装得是对应的纵向的偏移坐标）
    # 此时的shift变量就变成了以特征图像素总个数为行，4列的这样的数据格式（堆叠成四列是因为anchor的表示是左上右下坐标的形式，
    # 所有有四个坐标，而每两列恰好代表了横纵坐标的偏移量也就是一个点，所以最后用四列代表了两个点的偏移量。）
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1) # shift.shape = (height * width, 4)

    A = anchor_base.shape[0] # 读取anchor_base的个数,A=9
    K = shift.shape[0] # 读取特征图中元素的总个数

    # 用基础的9个anchor的分别和偏移量相加，得出所有anchor的坐标（四列可以看作是左上角的坐标和右下角的坐标加偏移量的同步执行。
    # 一共K个特征点，每个点有9个基本的anchor，reshape成((K*A),4)的形式，得到了最后的所有的anchor坐标）
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor #anchor.shape = (height * width, 4)


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    # 有个张量a，那么a.normal_()就表示用标准正态分布填充a，是in_place操作
    # torch里面所有带下划线_()的函数都是in_place操作
    if truncated:
        # PyTorch torch.fmod()方法给出除数除以元素的余数。除数可以是数字或张量
        
        # x.mul(y)或x.mul_(y)实现把x和y点对点相乘，其中x.mul_(y)是in-place操作，
        # 会把相乘的结果存储到x中。值得注意的是，x必须是tensor, y可以是tensor，也可以是数。

        # .add()和.add_()都能把两个张量加起来，但.add_是in-place操作，比如x.add_(y)，
        # x+y的结果会存储到原来的x中。Torch里面所有带"_"的操作，都是in-place的。
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
