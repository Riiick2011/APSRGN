# 本文件记录网络的配置参数
# Model configuration


yowo_config = {
    'yowo_v3_large': {
        'decoupled_position': '2DBackbone',  # 解耦发生的位置，可选2DBackbone、Neck、Nowhere

        ## Backbone
        # 2D    输出的通道数都是256，空间尺寸都是3个层级
        'backbone_2d': 'yolo_free_large',  # 'yolov8l', # 'yolo_free_large' 分别是解耦的2DBackbone和非解耦的2DBackbone
        'pretrained_2d': True,
        'stride': [8, 16, 32],  # 空间上的stride层级
        # 3D  输出的通道数不一样，空间尺寸是三个层级，时间尺寸是len_clip/4 /8 /16
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        'multilevel_3d': False,  # 开启后则3D骨架输出3个层级的特征图

        ## Neck
        'fpn': False,
        'fpn_after_encoder': False,  # 跟在编码器后面 只交换信息 不改变空间尺寸和通道数量
        'encoder_type': 'CFSAB',  # 默认是CFSAB，还可以选CBAB、CFPAB，PASS、PASS_3等代表无注意力操作仅简单拼接后通过卷积来融合双分支的通道
        'attention_type': [['ca', ]],  # 对于CFSAB和CFPAB有效，嵌套列表，第一层默认有1项，
        # 如果有2项则第2项代表reg_encoder；可以有ca, sa , mpca等，顺序有影响; 默认是[['ca', ]]、[['ca', 'sa'], ['ca', 'mpca']]

        ## Head  Head、Matcher、Loss的类别一般是一起换的
        'head_type': 'Headv2',  # Head类别，可选Headv2、Headv8
        # Headv2&Headv8
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'head_temporal': False,
        # Headv2
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # Headv8
        'reg_max': 8,  # 大于1则表示开启DFL分布式focal损失    表示预测框的lt 和 rb到 锚点中心的横向距离、纵向距离的上限-1(没乘以stride)

        ## Matcher
        # SimOTA&TAL
        'topk_candidate': 10,  # Max Positive Sample Number of one gt bbox 一个目标框最多得到10个正样本
        # SimOTA 与Headv2搭配
        'center_sampling_radius': 2.5,  # Positive Sample Radius of Grid Cell

        ## Loss
        # ConfCriterion&NoConfCriterion
        'loss_iou_type': 'giou',  # Conf default: 'giou',NoConf default：'ciou'
        'matcher_iou_type': 'iou',   # Conf default:'iou',NoConf default：'ciou'
        # ConfCriterion 与Headv2搭配
        'CCloss_conf_weight': 1,
        'CCloss_cls_weight': 1,
        'CCloss_box_weight': 5,
        'conf_iou_aware': False,   # conf分支是否关注iou  默认是False
        # NoConfCriterion 与Headv8搭配
        'cls_loss_type': 'BCE',  # cls_loss  BCE   应该进行实验
        'NCCloss_box_weight': 7.5,
        'NCCloss_cls_weight': 0.5,
        'NCCloss_dfl_weight': 1.5,
        # Track Loss
        'loss_track_weight': 0.1,
        'track_mix_ratio': 0.1,
        'tracknet_method': 'KalmanFilter',  # 可选 'Encoder'
        'track_attention': True,  # 可选 是否使用跟踪注意力模块计算置信度图

        ## NMS
        'nms_iou_type': 'iou',  # default：'iou'  如果修改要连同nms_thresh一起修改
        'nms_thresh': 0.5,  # 只对评估时有用
    },
}
