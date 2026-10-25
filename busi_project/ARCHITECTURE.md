# BUSI 实验架构

## 数据流

```text
BUSI 原始目录
  -> data.py 收集三类图像并排除 *_mask
  -> 固定随机种子，分层划分 70%/10%/20%
  -> runs/busi/split_data/{train,validation,test}
  -> ImageDataGenerator 读取与增强
  -> models.py 创建 DenseNet121/其他迁移学习模型
  -> train.py 类别加权训练、保存 checkpoint 和 CSV 日志
  -> evaluate.py 计算指标、保存 metrics.json 和混淆矩阵
```

## 文件职责

- `config.py`：数据目录、输出目录、图像尺寸、批大小和训练轮数。
- `data.py`：收集图像、过滤分割掩膜、分层划分数据、创建 Keras generator。
- `models.py`：创建 DNBCD（DenseNet121 + 两层全连接）和对照模型。
- `train.py`：PyCharm/命令行入口，执行训练、微调、保存模型和评估。
- `evaluate.py`：计算 Accuracy、Precision、Recall、F1、MAE、RMSE、AUC，并保存混淆矩阵。

## 与原 Notebook 的关系

原 BUSI Notebook 把所有步骤重复写在多个单元格中，并使用 Kaggle 路径。这个项目将路径参数化，并保留原 DNBCD 的两阶段训练：先冻结 DenseNet 主干，再用较小学习率解冻微调。对照模型可通过 `--model` 切换。

## 注意事项

BUSI 文件夹中通常同时有原图和分割掩膜。分类实验应排除文件名包含 `_mask` 的文件，否则掩膜会被当作独立样本，导致数据泄漏和结果失真。
