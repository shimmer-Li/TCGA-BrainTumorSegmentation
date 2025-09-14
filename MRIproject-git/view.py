import torch
from main  import test_loader
from Unet import Unet
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#加载训练好的模型
net = Unet()
net.load_state_dict(torch.load('model/best_model.pth', map_location=device))
net.to(device)
net.eval()

image,mask = next(iter(test_loader))
image,mask = image.to(device),mask.to(device)

pred_mask = net(image)
mask = torch.squeeze(mask)
#print(pred_mask.shape)             [8,2,256,256]
#print(mask.shape)                  [8,256,256]

num = 3
plt.figure(figsize=(10, 10))

for i in range(num):
    # 原图
    plt.subplot(num, 3, i * num+ 1)
    plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.title("Image")

    # 标签
    plt.subplot(num, 3, i * num + 2)
    plt.imshow(mask[i].cpu().numpy())
    plt.axis("off")
    plt.title("Mask")

    # 预测
    plt.subplot(num, 3, i * num + 3)
    pred = torch.argmax(pred_mask[i], dim=0).cpu().detach().numpy()
    plt.imshow(pred)
    plt.axis("off")
    plt.title("Predicted")
plt.show()
