# 基于LoRa-SAM的图像分割工具
---
**分割效果展示**

![demo](./demo/seg.png)

这是一个基于 [Segment Anything](https://github.com/facebookresearch/segment-anything) 和 [Improving the Generalization of Segmentation Foundation Model under Distribution Shift via Weakly Supervised Adaptation](https://github.com/zhang-haojie/wesam)论文模型的使用Streamlit搭建的图像分割web工具。

---

## 📁 项目结构
``` graphql
基于LoRa-SAM的图像分割工具
├─ demo
│  └─ seg.png           # 分割可视化演示图
├─ model                # 模型模块（含LoRA-SAM模型构建与推理逻辑）
│  ├─ adaptation.py     # LoRA-SAM 训练过程脚本
│  ├─ configs           # 模型和路径等基础配置文件夹
│  │  ├─ base_config.py
│  │  └─ config.py
│  ├─ datasets          # 数据集处理文件夹
│  ├─ INSTALL.md
│  ├─ LICENSE
│  ├─ losses.py         # 训练损失函数定义（自训练损失、对比损失、锚定损失）
│  ├─ model.py          # 主模型构建文件（集成LoRA和SAM）
│  ├─ PREPARE.md
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ sam_lora.py       # LoRA权重注入模块
│  ├─ segment_anything  # SAM源码
│  │  ├─ automatic_mask_generator.py   # 自动掩码生成器
│  │  ├─ build_sam.py   # 构建SAM模型的辅助函数
│  │  ├─ modeling
│  │  ├─ predictor.py   # 点/框 prompt 推理器
│  │  ├─ utils
│  │  └─ __init__.py
│  ├─ utils             # 相关工具
│  └─ validate.py       # 模型评估脚本
├─ README.md
├─ requirements.txt
├─ sam_st.py               # 前端主入口脚本（基于 Streamlit 实现交互）
├─ streamlit_dc            # streamlit_drawable_canvas前端文件夹
└─ util.py                 # 一些自定义通用工具函数

```

## How to run
### 1. install Modified streamlit_drawable_canvas
我们修改了 [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas) ，增加了 Left and Right Click 功能

如果你已经下载了`streamlit_drawable_canvas`, 请卸载它
```bash
pip uninstall streamlit-drawable-canvas
```
请下载我们修改后的版本（确保你已经下载了`npm`）
```bash
cd streamlit_dc/streamlit_drawable_canvas/frontend
npm install
npm run build
cd streamlit_dc/
pip install -e .
cd ../
```

### 2. install dependencies and get checkpoints
```shell
pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git
pip install -r requirements.txt

mkdir checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoint/sam_vit_b_01ec64.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoint/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoint/sam_vit_h_4b8939.pth
```

### 3. Run
```shell
streamlit run sam_st.py
```

---

## Thanks to

[segment anything](https://github.com/facebookresearch/segment-anything)

[wesam](https://github.com/zhang-haojie/wesam)

[streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
