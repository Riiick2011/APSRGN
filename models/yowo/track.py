"""
定义了跟踪交叉注意力模块
"""
import torch
import torch.nn as nn
from utils.utils import bbox_iou
from evaluator.kalman_filter import KalmanFilter
from models.basic.attention import TSAM


"""Track_Det和Track_Tube只用于 KalmanFilter方法"""


class Track_Det(object):
    def __init__(self, det, score):
        self.bbox = det  # x1y1x2y2
        self.score = score
        self.xyah = self.to_xyah()

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.bbox.copy()
        xc = (self.bbox[0] + self.bbox[2]) / 2
        yc = (self.bbox[1] + self.bbox[3]) / 2
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        ret = [xc, yc, width/height, height]
        return ret


class Track_Tube(object):
    def __init__(self, det):
        self.det_list = [det.bbox]  # 用列表表示,每一项是一个形状为(4,)数组，对应一个检测框
        self.active = True
        self.score_list = [det.score]
        self.tube_score = sum(self.score_list)/len(self.score_list)
        self.miss_link_times = 0

        # 卡尔曼滤波
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(det.xyah)

    def __call__(self, det):  # 当前时刻进行了关联，因此也要用当前时刻的测量值更新一次卡尔曼
        # 更新轨迹
        self.det_list.append(det.bbox)  # 每一个时刻最多有一个检测框
        self.score_list.append(det.score)
        self.tube_score = sum(self.score_list) / len(self.score_list)
        self.miss_link_times = 0  # 重置漏检次数

        # 用当前时刻的测量值更新卡尔曼滤波器
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, det.xyah, det.score)

    def predict(self):  # 每个时刻调用一次，预测下一时刻的位置，同时预测下一时刻的内部均值
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)  # 预测该时刻的均值和协方差
        xyah, _ = self.kf.project(self.mean, self.covariance)  # 变换回测量空间
        xc, yc, a, h = xyah
        w = a*h
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        xyxy = [x1, y1, x2, y2]

        return xyxy, self.tube_score  # 按理说不该取这个作为预测框的置信度，因为这样取的话，只要没有观测值更新，那么对未来任何时刻的预测的置信度都是相同的

    def miss_link(self):
        self.miss_link_times += 1
        if self.miss_link_times >= 5:  # 当漏检时长过长时，判定该管道死亡
            self.active = False


"""Track_Det和Track_Tube只用于 KalmanFilter方法"""


class Track_Block(nn.Module):  # 输入一个样本的先前检测框列表，输出该样本对当前时刻的检测框预测注意力图
    """ Track Block """
    def __init__(self, m_cfg):  # ch_in, kernels
        super().__init__()
        self.tracknet_method = m_cfg['tracknet_method']
        self.track_attention = m_cfg['track_attention']
        if self.tracknet_method == 'KalmanFilter':
            self.iou_threshold = 0.5
            self.act1 = nn.SiLU(inplace=True)
            self.TSAM = TSAM()
        elif self.tracknet_method == 'Encoder':
            self.act0 = nn.SiLU(inplace=True)  # 将label变换为0-1之间
            self.ln1 = nn.Linear(7, 64)
            self.act1 = nn.SiLU(inplace=True)  # 将不定长的第一维度，变换为1x64
            self.ln2 = nn.Linear(64, 224*224)
            self.act2 = nn.SiLU(inplace=True)  # 将不定长的第一维度，变换为1x64
            self.encoder = nn.Sequential(self.act0, self.ln1, self.act1, self.ln2, self.act2)

    def forward(self, x):
        """
            inputs :
                x : 一个2维度tensor，第一维长度不定，第二维长度为7，分别是时差、置信度、类别、两点百分比框
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        dtype = x.dtype
        device = x.device
        # 得到预测框
        if self.tracknet_method == 'KalmanFilter':
            x = x.detach().cpu().numpy()
            time_dif_max = int(x[:, 0].max())  # 该clip中距离当前帧最早的时刻的时差
            dets_clip = [[] for _ in range(time_dif_max)]

            for row in x:
                dets_clip[int(row[0])-1].append(Track_Det(row[-4:], row[2]))
            dets_clip = dets_clip[::-1]  # 最早的时刻排在最前面

            # 关联不同时差的检测框
            tubes_list = []
            for i in range(time_dif_max):  # 对于所有的时刻  关联时先考虑考虑置信度
                dets_frame = dets_clip[i]  # 该时刻的检测框
                active_tubes = (
                    sorted([tube for tube in tubes_list if tube.active],
                           key=lambda x: x.tube_score, reverse=True))  # 该时刻存活的管道，按照得分降序排列
                for active_tube in active_tubes:
                    pred_det, _ = active_tube.predict()  # 该时刻存活的管道对该时刻进行预测
                    if len(dets_frame) > 0:
                        ious_frame = []
                        for det_index in range(len(dets_frame)):
                            ious_frame.append(bbox_iou(pred_det, dets_frame[det_index].bbox))
                        if max(ious_frame) >= self.iou_threshold:
                            iou_max_index = ious_frame.index(max(ious_frame))  # 找到iou最大的索引
                            active_tube(dets_frame.pop(iou_max_index))  # 该管道关联该检测
                        else:
                            active_tube.miss_link()
                    else:
                        active_tube.miss_link()
                for det in dets_frame:  # 对于该时刻上剩余的检测框 新建管道
                    tubes_list.append(Track_Tube(det))

            # 对关键帧进行预测
            preds = []
            active_tubes = (
                sorted([tube for tube in tubes_list if tube.active],
                       key=lambda x: x.tube_score, reverse=True))  # 关键帧时刻存活的管道，按照得分降序排列
            for active_tube in active_tubes:
                pred_det, pred_score = active_tube.predict()
                pred_det.insert(0, pred_score)  # 长度为5
                preds.append(pred_det)
            preds = torch.tensor(preds, dtype=dtype, device=device)

            # 根据对当前帧预测的检测框，生成一张置信度图
            pred_num = len(preds)
            if pred_num:  # 防止没有预测框的情况
                Track_Conf_Map = torch.zeros((224, 224), device=device)
                confs = preds[:, 0]
                bboxes = torch.floor(preds[:, -4:] * 224).to(torch.int)
                if self.track_attention:  # 如果使用跟踪注意力
                    pred_maps = [torch.zeros((224, 224), device=device) for i in range(pred_num)]
                    for i in range(pred_num):  # 预测框数量
                        x1 = bboxes[i, 0]
                        y1 = bboxes[i, 1]
                        x2 = bboxes[i, 2]
                        y2 = bboxes[i, 3]
                        pred_maps[i][y1:y2, x1:x2] = confs[i]
                    pred_maps = torch.stack(pred_maps)  # pred_num,H,W
                    pred_maps = self.TSAM(pred_maps)   # pred_num,H,W
                    Track_Conf_Map = torch.sum(pred_maps, dim=0)  # H,W

                else:
                    for i in range(bboxes.shape[0]):
                        x1 = bboxes[i, 0]
                        y1 = bboxes[i, 1]
                        x2 = bboxes[i, 2]
                        y2 = bboxes[i, 3]
                        Track_Conf_Map[y1:y2, x1:x2] += confs[i]
            else:  # 如果没有预测框的
                Track_Conf_Map = torch.ones((224, 224), device=device)

            Track_Conf_Map = self.act1(Track_Conf_Map)  # 对所有的轨迹置信度图进行一次激活函数

        elif self.tracknet_method == 'Encoder':
            x = x.unsqueeze(0)
            x = self.act0(x)
            x = self.ln1(x)
            x = self.act1(x)
            x = torch.mean(x, dim=1, keepdim=True)

            x = self.ln2(x)
            track_enmbedding = self.act2(x)
            # track_enmbedding = self.encoder(x)  # N,7 -> N,64 -> 1,64 -> 1, 224x224
            Track_Conf_Map = track_enmbedding.reshape(224, 224).clamp(0, 1)  # 1,64 -> 1,7

        # 要预测的是趋势
        # 跟踪算法方式:
        # 图卷积方式: 剩余的作为节点计算当前帧节点位置
        # 卷积方式1: 构建一张tensor图并赋值，值由时差和置信度来计算而定，计算时的权重等待学习
        # 得到的注意力图要计算损失，损失的标注是将有真实框的区域标注为1，其余位置标注为0；或者

        return Track_Conf_Map  # 返回1张224x224的注意力图


class TrackNet(nn.Module):  # 输出到哪个位置也要实验，224x224的注意力图要通过平均池化转换为3种不同尺寸的注意力图
    """ Channel Fuse Series Attention Block """

    def __init__(self, m_cfg):  # ch_in, kernels
        super().__init__()
        self.track_mix_ratio = m_cfg['track_mix_ratio']
        self.tb1 = Track_Block(m_cfg)
        self.pool1 = nn.AvgPool2d(8, 8)
        self.pool2 = nn.AvgPool2d(16, 16)
        self.pool3 = nn.AvgPool2d(32, 32)
        self.pool = nn.ModuleList([self.pool1, self.pool2, self.pool3])

    def forward(self, x, feats):
        """
            inputs :
                x : 由Tensor构成的列表，每一个Tensor对应一个样本
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        B = len(x)
        Track_Conf_Maps = torch.ones((B, 224, 224)).to(x[0].device)
        for i, track_tensor in enumerate(x):  # 返回B张注意力图
            if len(track_tensor):
                Track_Conf_Maps[i] = self.tb1(track_tensor)

        if len(feats) == 2:
            decoupled = True
            cls_feats = feats[0]
            reg_feats = feats[1]
            cls_feats_ = []  # 储存不同等级下所有批次的tensor
            reg_feats_ = []
            for level in range(len(cls_feats)):
                cls_feats__ = []  # 储存一个等级下不同样本的tensor
                reg_feats__ = []
                for b in range(B):
                    att_map_for_this_level_and_sample = self.pool[level](Track_Conf_Maps[b].reshape(1, 1, 224, 224))

                    cls_feats__.append(
                            cls_feats[level][b, :, :, :] * (1-self.track_mix_ratio) +
                            cls_feats[level][b, :, :, :] * att_map_for_this_level_and_sample.squeeze(0).repeat(256, 1, 1) * self.track_mix_ratio)
                    reg_feats__.append(
                            reg_feats[level][b, :, :, :] * (1-self.track_mix_ratio) +
                            reg_feats[level][b, :, :, :] * att_map_for_this_level_and_sample.squeeze(0).repeat(256, 1, 1) * self.track_mix_ratio)
                cls_feats_.append(torch.stack(cls_feats__))
                reg_feats_.append(torch.stack(reg_feats__))

            return [cls_feats_, reg_feats_], Track_Conf_Maps

        else:
            decoupled = False
            feats_ = []  # 储存不同等级下所有批次的tensor
            for level in range(len(feats)):
                feats__ = []  # 储存一个等级下不同样本的tensor
                for b in range(B):
                    att_map_for_this_level_and_sample = self.pool[level](Track_Conf_Maps[b].reshape(1, 1, 224, 224))
                    feats__.append(
                            feats[level][b, :, :, :] * (1-self.track_mix_ratio) +
                            feats[level][b, :, :, :] * att_map_for_this_level_and_sample.squeeze(0).repeat(256, 1, 1) * self.track_mix_ratio)
                feats_.append(torch.stack(feats__))
            return feats_, Track_Conf_Maps
