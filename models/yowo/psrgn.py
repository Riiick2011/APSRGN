import torch
import torch.nn as nn

class PSRGN(nn.Module):
    def __init__(self):
        super().__init__()


def build_psrgn(head_type='head_v2'):
    psrgn=PSRGN()
    # 构建网络并返回一个子网络
    # 然后得到psrgn1和2，其中psrgn对于层级内部和层级之间进行
    return psrgn
    # 在inferrence和train中增加判断，如果是use_psrgn，则对于forward_once的多层级输出还要进行运算