# zfw_captcha_train

基于 CRNN（CNN + 双向 LSTM + CTC）的数字验证码识别项目，支持多种模型规格，方便在不同部署场景下选用合适大小的 `.pth` 权重文件。

> 🆕 **本次更新：训练监控全面接入 [SwanLab](https://swanlab.cn/)**，替换掉了原先的本地 TensorBoard + HTML 监控页面。指标、学习率、预测样图等都会推送到 SwanLab，无需再手动开 `8080` 端口或浏览器查看本地静态页。

---

## 1. 模型规格（`--variant`）

代码内置四档模型，对应不同的 `pth` 文件体积，供你按需训练 / 部署：

| variant   | 参数量      | 保存的 `.pth` 大小 | 适用场景                       |
| --------- | ----------- | -------------------- | ------------------------------ |
| `tiny`    | ~ 0.27 M    | **约 1 MB**          | 边缘设备 / 浏览器端 / 极致轻量 |
| `small`   | ~ 0.79 M    | **约 3 MB**          | 移动端 / 嵌入式               |
| `medium`  | ~ 2.56 M    | **约 10 MB**         | 普通服务端，速度与精度平衡     |
| `large`   | ~ 14.34 M   | **约 55 MB（无上限）** | 追求最高精度，使用 ResNet-18 主干 |

> 模型实现见 [`src/model.py`](src/model.py) 中的 `LightCRNN`（tiny/small/medium）与 `CRNN`（large），统一通过 `build_model(variant, num_classes)` 工厂函数构造。

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

最常见的用法 —— 训练一个约 1 MB 的 tiny 模型，并把指标推送到 SwanLab：

```bash
python src/train.py --variant tiny
```

训练其它规格：

```bash
python src/train.py --variant small      # ~3MB
python src/train.py --variant medium     # ~10MB
python src/train.py --variant large      # 不限大小（ResNet-18 主干）
```

训练结束后，权重文件会保存到：

```
checkpoints/<variant>/best_model.pth      # 验证集表现最好的 checkpoint（含优化器状态）
checkpoints/<variant>/final_model.pth     # 训练结束时的纯权重，可直接用于推理
```

`final_model.pth` 是「干净」的 `state_dict`，没有任何额外信息，文件大小就是表格里给出的目标体积。

### SwanLab 相关参数

| 参数                   | 说明                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `--swanlab-project`    | 项目名，默认 `zfw_captcha_train`                              |
| `--swanlab-experiment` | 本次实验名，默认 `<variant>-<时间戳>`                         |
| `--swanlab-workspace`  | 工作空间 / 组织名，可选                                       |
| `--swanlab-mode`       | `cloud` / `local` / `offline` / `disabled`，默认 `cloud`      |

例如：完全离线训练，但仍想保留训练日志：

```bash
python src/train.py --variant tiny --swanlab-mode offline
```

不希望使用 SwanLab 时：

```bash
python src/train.py --variant tiny --swanlab-mode disabled
```

会被记录到 SwanLab 的内容包括：

- `train/loss`、`train/accuracy`、`val/loss`、`val/accuracy`
- 学习率 `learning_rate`
- 每隔若干 epoch 的样例预测图（`val/predictions`）
- 完整的运行配置（变体、batch_size、参数量等）

### 其它常用参数

| 参数              | 说明                                              |
| ----------------- | ------------------------------------------------- |
| `--resume PATH`   | 从某个 checkpoint 继续训练                       |
| `--no-pretrained` | 训练 `large` 时跳过下载 ImageNet 预训练权重      |
| `--seed`          | 随机种子                                          |
| `--nodes` / `--node-rank` / `--dist-url` | 多机 DDP 分布式训练参数 |

> `large` 变体使用 `torchvision` 的 ResNet-18 ImageNet 预训练权重做初始化；如果机器无法访问外网，代码会自动回退到随机初始化，也可以显式加 `--no-pretrained`。

---

## 5. 评估

```bash
python src/evaluate.py --variant tiny
# 或指定权重路径
python src/evaluate.py --variant small --model-path checkpoints/small/final_model.pth
```

输出会打印对应模型在验证集上的整体准确率（按整张验证码完全正确为标准）。

---

## 6. 项目结构

```
.
├── config.py                # 全局配置：数据路径、超参、变体列表、SwanLab 设置
├── requirements.txt
├── src/
│   ├── model.py             # CRNN / LightCRNN + build_model() 工厂
│   ├── dataset.py           # 数据集加载（含 DDP 采样器）
│   ├── train.py             # 训练入口（接入 SwanLab）
│   ├── evaluate.py          # 评估脚本
│   └── utils.py             # 编解码 / checkpoint / 可视化等辅助函数
└── checkpoints/<variant>/   # 训练产物
```

---

## 7. 与上一版的主要差异

- **监控平台**：移除了 `src/web_monitor.py`、`src/monitor.py`、`monitor_template.html` 与 TensorBoard，统一改用 SwanLab。
- **模型选择**：新增 `--variant` 参数，提供 `tiny / small / medium / large` 四档大小。
- **权重产物**：每个 variant 的 checkpoint 各自分目录，`final_model.pth` 严格对应 1 MB / 3 MB / 10 MB / 不限大小。
- **离线友好**：`large` 模型在无网环境下会自动回退到随机初始化，不再因为下载预训练权重失败而中断训练。
