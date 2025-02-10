# Universal Domain Adaptive Object Detection via Dual Probabilistic Alignment (DPA)
Code implementation for Universal Domain Adaptive Object Detection via Dual Probabilistic Alignment (DPA)

![LAST COMMIT](https://img.shields.io/github/last-commit/zyfone/UniDAOD) 
![ISSUES](https://img.shields.io/github/issues/zyfone/UniDAOD)
![STARS](https://img.shields.io/github/stars/zyfone/UniDAOD)



<div align="center">
  <img src="src\framework.png" alt="Framework" style="width: 100%; max-width: 450px;"/>
</div>


## Requirements

* Ubuntu 18.04.5 LTS
* Python 3.6
* [CUDA 10.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.0.0](https://pytorch.org)
* [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

## Compile the code

```bash
#Compile the cuda dependencies using following simple commands following [Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0):
cd lib
python setup.py build develop
```

## Pre-trained Models


* **ResNet101:**  [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)



## Components

### Global-level Domain Private Alignment (GDPA)

- **Path:** `UniDAOD-DPA/lib/model/utils`
- **Function:** `global_alignment()`

### Instance-level Domain Shared Alignment (IDSA)

- **Path:** `UniDAOD-DPA/lib/model/da_faster_rcnn/`
- **File:** `openset_weight.py`

### Private Class Constraint (PCC)

- **Path:** `UniDAOD-DPA/lib/model/utils`
- **Function:** instance_alignment_private



## Training and Test

Train the model

```bash
CUDA_VISIBLE_DEVICES=0 python -u da_train_net.py \
--max_epochs 10 --cuda --dataset voc2clipart_0.25 \
--net res101 --save_dir ./weight_model/voc2clipart_0.25 \
--pretrained_path XXXX/pretrained_model/resnet101_caffe.pth \
--gc --lc --da_use_contex --weight_consis 0.1 --lr_bound 0.1 --gmm_split 0.03
```

Test the well-trained model:
```bash
python test_clipart_0.25.py >> test-voc025.out
```

Train the model and test the well-trained model through the script:
```bash
sh train_scripts\train_voc2clipart_0.25.sh
```


If you have any questions , please contact me at 478756030@qq.com


```BibTeX
@article{zheng2024universal,
  title={Universal Domain Adaptive Object Detection via Dual Probabilistic Alignment},
  author={Zheng, Yuanfan and Wu, Jinlin and Li, Wuyang and Chen, Zhen},
  journal={arXiv preprint arXiv:2412.11443},
  year={2024}
}
```