import numpy as np
from PIL import Image
from CamVid_dataloader11 import mask_to_class, Cam_COLORMAP

# 替换成你的标签图路径
label_path = "./CamVid/train_labels/0001TP_006690_L.png"
label = Image.open(label_path).convert('RGB')


#label_path = "./CamVid/train/0001TP_006690.png"
#label = Image.open(label_path).convert('RGB')

print(label.mode)
label = np.array(label)



# 转换标签
label_class = mask_to_class(label)

# 查看结果
print("标签图中出现的类别索引：", np.unique(label_class))
print("每个类别的像素数量：", np.bincount(label_class.flatten()))
print("颜色映射表：", Cam_COLORMAP)