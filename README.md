# zfw_captcha_train

纯 CNN 的 4 位数字验证码识别项目（针对学校自服务平台），使用 4 个独立分类头（每位数字一个），训练采用 CrossEntropyLoss。支持多种模型规格，方便按部署场景选用合适大小的 `.pth` 权重文件。

本项目引入了swanlab监测，训练情况见链接 [Swanlab zfw_captcha_training](https://swanlab.cn/@nancunchild/zfw_captcha_train?utm_source=website_qr&utm_medium=qr_scan)

> **设计决策**：验证码为固定 4 位、位置基本固定的纯数字，不存在变长对齐问题，因此不需要 RNN/CTC。早期版本曾使用 CRNN + CTC，但引入了不必要的复杂度和数值稳定性问题，已在 `3aeb432` 中移除。

> **训练监控**：全面接入 [SwanLab](https://swanlab.cn/)，指标、学习率、预测样图等都会推送到 SwanLab。

> **其他功能**：早停（`--patience`）、一次训练多个变体（`--variants`）、双卡并行训练多变体（`--parallel-variants`）、TF32 / Fused Adam / DDP 优化等。

---

## 1. 模型规格（`--variant`）

代码内置三档模型，对应不同的参数量和 `pth` 文件体积：

| variant   | 参数量      | 保存的 `.pth` 大小 | 适用场景                       |
| --------- | ----------- | -------------------- | ------------------------------ |
| `nano`    | ~ 21K       | **约 85 KB**         | 极致压缩（< 100 KB 硬限制）    |
| `small`   | ~ 96K       | **约 385 KB**        | 边缘设备 / 浏览器端 / 轻量部署（默认） |
| `full`    | ~ 196K      | **约 785 KB**        | 追求最高精度                   |

> 模型实现见 [`src/model.py`](src/model.py) 中的 `CaptchaCNN` 类，通过 `build_model(variant)` 工厂函数构造。架构为多层 3×3 Conv + BN + ReLU，经 `AdaptiveAvgPool2d((1, 4))` 坍缩为 4 列特征，每列接一个独立线性分类头输出 10 类（0-9）。

---

## 2. 环境准备

```bash
pip install -r requirements.txt
```

`requirements.txt` 已经把原来的 `tensorboard` 替换成了 `swanlab`。

登录 SwanLab（首次使用，按提示粘贴 API Key 即可，参考 https://docs.swanlab.cn/ ）：

```bash
swanlab login
```

如不希望联网上传，可在训练时改用本地或离线模式（见下文 `--swanlab-mode`）。

---

## 3. 数据准备

把验证码图片放到 `data/captcha_get/<label>/xxx.png`，目录名即为该图片的真实标签，例如：

```
data/captcha_get/
├── 0123/
│   ├── 0001.png
│   └── 0002.png
├── 4567/
│   ├── 0001.png
│   └── ...
└── ...
```

图像尺寸默认按 `90 × 34` 处理（详见 `config.py` 的 `IMG_WIDTH` / `IMG_HEIGHT`）。

---

## 4. 训练

### 单变体

```bash
python src/train.py --variant nano      # ~85KB
python src/train.py --variant small     # ~385KB (默认)
python src/train.py --variant full      # ~785KB
```

权重保存到：

```
checkpoints/<variant>/best_model.pth      # 验证集表现最好的 checkpoint（含优化器状态）
checkpoints/<variant>/final_model.pth     # 训练结束时的纯权重，可直接用于推理
```

`final_model.pth` 是「干净」的 `state_dict`，没有任何额外信息，文件大小就是表格里给出的目标体积。

### 一次训练多个变体（一行命令出多个产物）

```bash
# 顺序训练：每个变体依次训练，每个都吃满全部 GPU（DDP）
python src/train.py --variants nano,small,full

# 等价写法
python src/train.py --variants all

# 双卡并行：每个变体独占一张 GPU 同时训练
python src/train.py --variants all --parallel-variants
```

两种模式对比（双卡 + 3 变体）：

| 模式 | 命令开关 | 单变体训练时间 | 总耗时（粗估） | 备注 |
| ---- | --------- | -------------- | --------------- | ---- |
| 顺序 | 默认 | T / 1.8（DDP 加速） | **≈ 1.67 T** | 每个变体都用全部 GPU |
| 并行 | `--parallel-variants` | T（单卡） | **≈ 2 T** | 每变体独占 1 张卡 |

> 变体数 ≤ GPU 数时（如 3 个变体 + 4 张卡），`--parallel-variants` 可以真正同时跑完。

### 早停（`--patience`）

```bash
python src/train.py --variant small --patience 8       # 默认值
python src/train.py --variant small --patience 0       # 关闭早停
```

逻辑：如果验证集准确率连续 N 个 epoch 没刷新最高值，立即结束训练（DDP 模式下会广播停止信号给所有 rank）。`best_model.pth` 永远保留历史最优。

### 性能优化开关

下列参数默认开启常用加速；如需排除影响可手动关闭：

| 参数              | 默认 | 说明                                           |
| ----------------- | ---- | ---------------------------------------------- |
| `--num-workers`   | 4    | DataLoader worker 数（每进程）                |
| `--prefetch-factor` | 4  | 每个 worker 预取的 batch 数                   |
| TF32 矩阵乘      | 开 | Ampere+ GPU 上自动启用，关闭加 `--no-tf32`   |
| `cudnn.benchmark` | 开 | 加 `--deterministic` 改为 deterministic 模式（更慢但可复现） |
| Fused Adam        | 开 | PyTorch ≥ 2.0 + CUDA 自动启用，无 flag        |
| DDP `static_graph` | 开 | 加 `--no-static-graph` 关闭                  |
| `gradient_as_bucket_view` | 开 | 减少 DDP 梯度同步内存拷贝（默认 hardcoded） |
| `--sync-bn`       | 关 | 转 SyncBatchNorm，per-card batch 较小时建议开 |
| DataLoader `persistent_workers` | 开 | 跨 epoch 保留 worker 进程（自动启用）   |

---

## 5. SwanLab 相关参数

| 参数                   | 说明                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `--swanlab-project`    | 项目名，默认 `zfw_captcha_train`                              |
| `--swanlab-experiment` | 本次实验名，默认 `<variant>-<时间戳>`                         |
| `--swanlab-workspace`  | 工作空间 / 组织名，可选                                       |
| `--swanlab-mode`       | `cloud` / `local` / `offline` / `disabled`，默认 `cloud`      |

例如：完全离线训练，但仍想保留训练日志：

```bash
python src/train.py --variant small --swanlab-mode offline
```

不希望使用 SwanLab 时：

```bash
python src/train.py --variant small --swanlab-mode disabled
```

会被记录到 SwanLab 的内容包括：

- `train/loss`、`train/accuracy`、`val/loss`、`val/accuracy`
- 学习率 `learning_rate`
- 每隔若干 epoch 的样例预测图（`val/predictions`）
- 完整的运行配置（变体、batch_size、参数量、patience 等）

> 多变体训练时，每个变体会创建一个独立的 SwanLab run（命名为 `<variant>-<时间戳>`），可在 SwanLab 项目首页对比。

---

## 6. 评估

```bash
python src/evaluate.py --variant small
# 或指定权重路径
python src/evaluate.py --variant full --model-path checkpoints/full/final_model.pth
```

输出会打印对应模型在验证集上的整体准确率（按整张验证码完全正确为标准）。

---

## 7. 项目结构

```
.
├── config.py                # 全局配置：数据路径、超参、变体列表、SwanLab 设置
├── requirements.txt
├── src/
│   ├── model.py             # CaptchaCNN + build_model() 工厂（纯 CNN，4 分类头）
│   ├── dataset.py           # 数据集加载（含 DDP 采样器、persistent_workers）
│   ├── train.py             # 训练入口（接入 SwanLab，支持多变体并行/顺序）
│   ├── evaluate.py          # 评估脚本
│   └── utils.py             # 编解码 / checkpoint / 可视化等辅助函数
└── checkpoints/<variant>/   # 训练产物
```

---

## 8. 常见问题排查

- **训练 loss 一开始就是 NaN**：代码已内置非有限 loss 跳过机制和梯度裁剪（max_norm=5.0）。如仍出现，请检查标签是否包含非数字字符或长度不为 4。
- **CUDA out of memory**：把 `config.BATCH_SIZE` 调小，或切到更小的 variant。
- **多卡训练效率不高**：先确认 `nvidia-smi` 看 GPU 利用率；可尝试 `--num-workers 8 --prefetch-factor 8`，或加 `--sync-bn`。

---

## 9. 架构演进记录

| 阶段 | 架构 | 损失函数 | 问题 |
|------|------|----------|------|
| v1 (初始) | CRNN (CNN + BiLSTM + CTC) | CTCLoss | CTC NaN、blank token 处理复杂、对固定长度任务过度设计 |
| v2 (当前) | 纯 CNN + 4 独立分类头 | CrossEntropyLoss × 4 | 无 |

**主要改动**：
- **架构简化**：移除 RNN/LSTM 和 CTC，改为 AdaptiveAvgPool + 多头分类，代码量和 debug 难度大幅下降。
- **监控平台**：移除了 TensorBoard / 本地 HTML 监控，统一改用 SwanLab。
- **模型选择**：提供 `nano / small / full` 三档大小。
- **多产物**：`--variants` 一次训多个变体，可选 `--parallel-variants` 多卡并发。
- **早停**：`--patience` 默认 8 epoch。
- **性能**：默认开启 TF32 / cudnn.benchmark / Fused Adam / DDP static_graph / persistent_workers。
- **数值稳定性**：梯度裁剪 5.0、跳过非有限 loss 的 batch。
