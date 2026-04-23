import os
import glob
import torch
import numpy as np
import time
from torchvision import transforms
from PIL import Image

# 加载模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

from model.unet import UNet
from model.model import SelfNet


model_selection = 'SelfNet' # UNet,SelfNet

if model_selection == 'UNet':
    model = UNet(in_channels=3, num_classes=4).to(device)
    output_dir = './predict_output/baseline_predictions/'   
if model_selection == 'SelfNet':
    model = SelfNet(in_channels=3, num_classes=4).to(device)
    output_dir = './predict_output/test_predictions/'   
save_pth_path = model_selection+'_pth'



# model = UNet(in_channels=3, num_classes=4).to(device)
model.load_state_dict(torch.load('./{}_pth/best_model.pth'.format(model_selection)))  
model.eval()


# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
])


# 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)
    return prediction.cpu().numpy()[0]

# 保存预测结果为 NPY 文件
def save_prediction_as_npy(predicted_mask, save_path):
    np.save(save_path, predicted_mask)


# 设置路径
input_dir = './dataset/images_test/'               

# 在处理所有图像之前，记录开始时间
start_time = time.time()

# 遍历输入目录中的所有图像文件
for image_path in glob.glob(os.path.join(input_dir, '*.*')):
    predicted_mask = predict_image(image_path)

    # 保存预测结果为 NPY，格式为 prediction_XXX.npy
    base_name = os.path.basename(image_path)  # 获取原始图像名称
    index_jpg = base_name.split('.')[0]
    output_npy_path = os.path.join(output_dir, f'prediction_{index_jpg}.npy')
    save_prediction_as_npy(predicted_mask, output_npy_path)
    print(f'预测结果已保存为: {output_npy_path}')

# 记录结束时间
end_time = time.time()

# 计算推理时间和FPS
total_time = end_time - start_time
num_images = len(glob.glob(os.path.join(input_dir, '*.*')))
average_fps = num_images / total_time

print('---------------------')
print('model:',model_selection)
print(f'推理总时间: {total_time:.2f}秒')
print(f'平均每张图像推理FPS: {average_fps:.2f}')


