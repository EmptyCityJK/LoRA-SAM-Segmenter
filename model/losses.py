import random  # 导入随机数模块
import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用的函数式API
from abc import ABC  # 导入抽象基类

ALPHA = 0.8  # Focal Loss的alpha参数，控制正负样本平衡
GAMMA = 2    # Focal Loss的gamma参数，控制难易样本关注度

# 焦点损失函数：适用于类别极度不平衡的分类或分割任务（如医学图像分割、目标检测等）
class FocalLoss(nn.Module):  # 定义Focal Loss损失函数类，继承自nn.Module

    def __init__(self, weight=None, size_average=True):
        super().__init__()  # 调用父类初始化方法

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)  # 对输入做sigmoid，得到概率
        inputs = torch.clamp(inputs, min=0, max=1)  # 限制概率在[0,1]之间
        #flatten label and prediction tensors
        inputs = inputs.view(-1)  # 将输入展平成一维
        targets = targets.view(-1)  # 将目标展平成一维

        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')  # 计算逐元素二元交叉熵
        BCE_EXP = torch.exp(-BCE)  # 计算e^(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE  # Focal Loss公式
        focal_loss = focal_loss.mean()  # 求均值

        return focal_loss  # 返回损失

# 骰子损失函数：适用于医学图像分割任务，衡量预测和真实分割掩码之间的相似度
class DiceLoss(nn.Module):  # 定义Dice Loss损失函数类，继承自nn.Module

    def __init__(self, weight=None, size_average=True):
        super().__init__()  # 调用父类初始化方法

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)  # 对输入做sigmoid
        inputs = torch.clamp(inputs, min=0, max=1)  # 限制概率在[0,1]之间
        #flatten label and prediction tensors
        inputs = inputs.view(-1)  # 将输入展平成一维
        targets = targets.view(-1)  # 将目标展平成一维

        intersection = (inputs * targets).sum()  # 计算交集
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  # 计算Dice系数

        return 1 - dice  # 返回1-Dice作为损失

# 对比损失：用于度量两个嵌入向量之间的相似性，常用于度量两个图像之间的相似性
class ContraLoss(nn.Module):  # 定义对比损失（Contrastive Loss）类，继承自nn.Module

    def __init__(self, temperature = 0.3, weight=None, size_average=True):
        super().__init__()  # 调用父类初始化方法
        self.temperature = temperature  # 温度参数，控制softmax平滑度
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失（未实际用到）

    def forward(self, embedd_x: torch.Tensor, embedd_y: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor):
        x_embedding = self.norm_embed(embedd_x) # embedd_x: [256, 64, 64]，对embedd_x做归一化
        y_embedding = self.norm_embed(embedd_y) # 对embedd_y做归一化

        # 将mask_x和mask_y插值到embedding的空间分辨率
        x_masks = F.interpolate(mask_x, size=x_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        y_masks = F.interpolate(mask_y, size=y_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()

        x_masks = F.sigmoid(x_masks)  # 对mask做sigmoid
        x_masks = torch.clamp(x_masks, min=0, max=1)  # 限制在[0,1]
        x_masks = x_masks > 0.5  # 二值化
        y_masks = F.sigmoid(y_masks)  # 对mask做sigmoid
        y_masks = torch.clamp(y_masks, min=0, max=1)  # 限制在[0,1]
        y_masks = y_masks > 0.5  # 二值化

        # x_masks = self.add_background(x_masks)
        # y_masks = self.add_background(y_masks)

        sum_x = x_masks.sum(dim=[-1, -2]).clone()  # 每个mask的像素和
        sum_y = y_masks.sum(dim=[-1, -2]).clone()  # 每个mask的像素和
        sum_x[sum_x[:, 0] == 0.] = 1.  # 防止除零
        sum_y[sum_y[:, 0] == 0.] = 1.  # 防止除零

        multi_embedd_x = (x_embedding * x_masks).sum(dim=[-1, -2]) / sum_x  # 计算每个mask区域的特征均值 [n, 256, 64, 64] -> [n, 256]
        multi_embedd_y = (y_embedding * y_masks).sum(dim=[-1, -2]) / sum_y  # 计算每个mask区域的特征均值

        flatten_x = multi_embedd_x.view(multi_embedd_x.size(0), -1)         # [n, 256] 展平成二维
        flatten_y = multi_embedd_y.view(multi_embedd_y.size(0), -1)         # [n, 256] 展平成二维
        # similarity_matrix = torch.matmul(multi_embedd_x, multi_embedd_y.T)
        similarity_matrix = F.cosine_similarity(flatten_x.unsqueeze(1), flatten_y.unsqueeze(0), dim=2)  # [n, n] 计算两两余弦相似度

        label_pos = torch.eye(x_masks.size(0)).bool().to(embedd_x.device)  # 对角线为正样本
        label_nag = ~label_pos  # 其余为负样本

        similarity_matrix = similarity_matrix / self.temperature    # 除以温度参数
        loss = -torch.log(
                similarity_matrix.masked_select(label_pos).exp().sum() / 
                similarity_matrix.exp().sum()
            )  # 计算对比损失
        # loss = -torch.log(
        #         similarity_matrix.masked_select(label_pos).exp().sum()
        #     )
        return loss  # 返回损失

    def norm_embed(self, embedding: torch.Tensor):
        embedding = F.normalize(embedding, dim=0, p=2)  # 对embedding做L2归一化
        return embedding

    def add_background(self, masks):
        mask_union = torch.max(masks, dim=0).values  # 取所有mask的并集
        mask_complement = ~mask_union  # 取反，得到背景
        concatenated_masks = torch.cat((masks, mask_complement.unsqueeze(0)), dim=0)  # 拼接背景
        return concatenated_masks