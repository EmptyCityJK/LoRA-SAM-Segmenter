import os
import cv2
import torch
from configs.config import cfg
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

from box import Box
from tqdm import tqdm
from model import Model
from datasets.PascalVOC import PascalVOCDataset
from datasets.COCO import COCODataset


def draw_image(image, masks, boxes, labels, alpha=0.4):
    '''可视化绘制'''
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def visualize(cfg: Box):
    '''主可视化函数'''
    # 初始化模型 - 这里需要根据你的Model类实现进行调整
    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()
    # 加载数据集
    dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
                          annotation_file=cfg.dataset.val.annotation_file,
                          transform=None)
    # dataset = PascalVOCDataset(cfg=cfg, root_dir=cfg.dataset.val.root_dir)
    # 获取预测器
    predictor = model.get_predictor()
    os.makedirs(cfg.out_dir, exist_ok=True)
    # 遍历所有图像
    for image_id in tqdm(dataset.image_ids):
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image_output_path = os.path.join(cfg.out_dir, image_info['file_name'])
        # 读取和处理图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 获取标注信息
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        bboxes = []
        # 提取边界框
        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])
        # 转换为张量
        bboxes = torch.as_tensor(bboxes, device=model.model.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        # 预测分割掩码
        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        # 生成可视化结果并保存
        image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
        cv2.imwrite(image_output_path, image_output)

if __name__ == "__main__":
    visualize(cfg)

