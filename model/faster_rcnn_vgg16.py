from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30] # 模型前30层
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6] #TODO:啥？
    if not opt.use_drop:
        del classifier[5] # 在ROIHead中删除dropout层
        del classifier[2] # 在ROIHead中删除dropout层
    # nn.Sequential 一个有序的容器，神经网络模块将按照在传入构造器的顺序
    # 依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv # TODO: why?
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        # 感觉https://www.cnblogs.com/king-lps/p/8995412.html 图上的fc6/fc7都是classifier部分
        # 是的，详见vgg-16的框架图
        self.classifier = classifier 
        self.cls_loc = nn.Linear(4096, n_class * 4) # head里面的最下面的FC_84网络，进4096，出84
        self.score = nn.Linear(4096, n_class) # head里面的最下面的FC_21网络，进4096，出21

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        # torchvision.ops.roi_pool(input, boxes, output_size, spatial_scale=1.0)
        # input (Tensor[N, C, H, W]) – 输入张量
        # boxes (Tensor[K, 5] or List[Tensor[L, 4]]) – 输入的box 坐标，格式：list(x1, y1, x2, y2) 或者(batch_index, x1, y1, x2, y2)
        # output_size (int or Tuple[int, int]) – 输出尺寸, 格式： (height, width)
        # spatial_scale (float) – 将输入坐标映射到box坐标的尺度因子. 默认: 1.0
        # ROIPool就是：根据rois，在featrues上对每个roi（兴趣区域，目标框）maxpool出一个7*7的池化结果
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale) #TODO: 更详细说明?

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        # cat()函数目的： 在给定维度上对输入的张量序列seq 进行连接操作。
        # outputs = torch.cat(inputs, dim=?) → Tensor
        # inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
        # dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。
        # python增加数组和减少数组维度的方法(None、np.newaxis和0的用法）
        # 增加数组维度,可以使用None(可以理解为New One)或者np.newaxis
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        # TODO: rois的序列调整之前为啥是yx，还有为啥要翻转
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，但是两个tensor完全没有联系
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool) # VGG-16的classifier()包含了两块FC_4096，参见VGG-16框架图
        roi_cls_locs = self.cls_loc(fc7) # FC_84 输入4096，输出84
        roi_scores = self.score(fc7) # FC_21 输入4096，输出21
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    # 在对VGG16RoIHead网络的全连接层权重初始化过程中，按照图像是否为truncated分了两种初始化分方法
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
