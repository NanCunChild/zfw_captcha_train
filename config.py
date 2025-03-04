# 数据集路径
DATA_DIR = 'data/captcha_get'

# 图像大小
IMG_WIDTH = 90
IMG_HEIGHT = 34

# 字符集
CHARS = '0123456789'
NUM_CHARS = len(CHARS)

# 训练参数
BATCH_SIZE = 32
EPOCHS = 50 #可以先尝试50轮，然后根据结果调整。
LEARNING_RATE = 0.001
DEVICE = 'cuda'  # 或 'cpu'

# 模型保存路径
MODEL_PATH = 'checkpoints/model.pth'