from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    # 从 enumerate(dataloader)里面依次读取数据，读取的内容是: imgs图片，sizes尺寸，
    # gt_boxes真实框的位置 gt_labels真实框的类别以及gt_difficults这些
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)): # tqdm模块是python进度条库
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # 利用faster_rcnn.predict(imgs,[sizes]) 得出预测的pred_boxes_,pred_labels_,pred_scores_预测框位置，
        # 预测框标记以及预测框的分数等等！这里的predict是真正的前向传播过程！完成真正的预测目的！
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters() # 在可视化界面重设所有数据
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda() # 将img,bbox,label,scale全部设置为可gpu加速
            trainer.train_step(img, bbox, label, scale) # 调用trainer.py中的函数trainer.train_step(img,bbox,label,scale)进行一次参数迭代优化过程

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file): # 如果达到判断debug_file是否存在，用ipdb工具设置断点
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data()) # 将训练数据读取并上传完成可视化

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                # 调用trainer.vis.img('pred_img',pred_img)将迭代读取原始数据中的原图，
                # bboxes框架，labels标签在可视化工具下显示出来
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                # 利用同样的方法将原始图片以及边框类别的预测结果同样在可视化工具中显示出来！
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                # 将rpn_cm也就是RPN网络的混淆矩阵在可视化工具中显示出来
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # 将Roi_cm将roi的可视化矩阵以图片的形式显示出来
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        # 将测试数据调用eval()函数进行评价，存储在eval_result中
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        # 将eval_result['map']在可视化工具中进行显示
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        # 用if判断语句永远保存效果最好的map！
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        # if判断语句如果学习的epoch达到了9就将学习率*0.1变成原来的十分之一
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        # 判断epoch==13结束训练验证过程
        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
