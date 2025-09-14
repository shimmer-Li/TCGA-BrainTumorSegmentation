import os
import glob
from PIL import Image
import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

from Unet import Unet
from torch.utils.tensorboard import SummaryWriter

'''————————创建数据集————————'''
kaggle_3m = './data/kaggle_3m/'
dirs = glob.glob(os.path.join(kaggle_3m, "*"))  #读取kaggle_3m文件夹下的所有文件
#print(dirs)
#print(len(dirs))

#第一次
data_img=[]
data_label=[]
for subdir in dirs:
    #dirname = subdir.split('\\')[-1]        #获取每一个子文件夹的目录
    for filename in os.listdir(subdir):
        img_path = os.path.join(subdir, filename)         #图片的绝对路径
        if 'mask' in img_path:
            data_label.append(img_path)
        else:
            data_img.append(img_path)

data_imgx=[]                                     #一一对应
for i in range(len(data_label)):
    img_mask = data_label[i]
    img = img_mask[:-9]+'.tif'
    data_imgx.append(img)

#寻找有病灶的图片
data_newimg=[]
data_newlabel=[]
for i in data_label:
    value = np.max(cv2.imread(i))                  #将图片转换成数组,寻找有病灶的图片
    try:
        if value > 0:
            data_newlabel.append(i)
            i_img = i[:-9]+'.tif'
            data_newimg.append(i_img)
    except:
        pass

#test
'''
im = Image.open(data_newimg[300])
im.show()
im_label = Image.open(data_newlabel[300])
im_label.show()
'''


'''————————划分训练集与测试集————————'''
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class BrainMRIdataset(Dataset):
    def __init__(self, img, label , transform):
        self.img = img
        self.label = label
        self.transform = transform
    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]

        img = Image.open(img)
        img = self.transform(img)

        label = Image.open(label)
        label = self.transform(label)
        label = torch.squeeze(label).type(torch.long)         #以适用CrossEntropyLoss

        return img, label
    def __len__(self):
        return len(self.img)

#print(len(data_newimg))
#print(len(data_newlabel))
s=1000
train_img = data_newimg[:s]
train_label = data_newlabel[:s]
test_img = data_newimg[s:]
test_label = data_newlabel[s:]

train_dataset = BrainMRIdataset(train_img, train_label, train_transform)
test_dataset = BrainMRIdataset(test_img, test_label, test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

'''
#test
img,label = next(iter(train_loader))
plt.figure(figsize=(12,8))
for i,(img,label) in enumerate(zip(img[:4],label[:4])):
    img = img.permute(1,2,0).numpy()
    label = label.numpy()
    plt.subplot(2,4,i+1)
    plt.imshow(img)
    plt.subplot(2,4,i+5)
    plt.imshow(label)
plt.show()
'''


'''———————开始训练————————'''
def dice_coefficient(pred, target, smooth=1e-6):
    # pred, target: (N, H, W) 0/1
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    # pred, target: (N, H, W) 0/1
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds shape: [B, C, H, W]   (网络输出，还没 argmax)
        # targets shape: [B, H, W]    (标签，0/1)

        preds = torch.softmax(preds, dim=1)[:, 1, ...]  # 取前景通道
        targets = targets.float()

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        return 1 - dice


def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Unet()
    net.to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    dice_loss = DiceLoss()

    def criterion(preds, targets):
        return 0.2 * loss_fn(preds, targets) + 0.8  * dice_loss(preds, targets)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 学习率调度器：当 val_dice 停滞时，lr 降一半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    max_epoch = 100
    patience = 20
    best_dice = 0.0
    no_improve = 0

    #训练
    total_train_step = 0
    total_test_step = 0

    writer = SummaryWriter("MRI")

    for i in range(1,max_epoch+1):
        print("————————第{}轮训练开始————————".format(i))

        #训练步骤
        net.train()
        for data in train_loader:
            imgs, labels = data
            imgs,labels = imgs.to(device), labels.to(device)
            outputs = net(imgs)
            loss = criterion(outputs, labels)

            #优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
                writer.add_scalar("Loss/train", loss.item(), total_train_step)

        #每结束一轮训练后，测试一次
        net.eval()
        total_test_loss = 0
        total_dice = 0
        total_iou = 0
        num_batches = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, labels = data
                imgs,labels = imgs.to(device), labels.to(device)
                outputs = net(imgs)
                preds = torch.argmax(outputs, dim=1)


                loss =criterion(outputs, labels)
                dice = dice_coefficient(preds, labels)
                iou = iou_score(preds, labels)

                total_test_loss += loss.item()
                total_dice += dice.item()
                total_iou += iou.item()
                num_batches += 1

        # 平均值
        avg_test_loss = total_test_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches


        total_test_step += 1
        print("第{}轮模型的Loss：{}".format(i+1,avg_test_loss))
        print("第{}轮模型的Dice：{}".format(i+1, avg_dice))
        print("第{}轮模型的IOU：{}".format(i+1, avg_iou))
        writer.add_scalar("Loss/test", avg_test_loss, total_test_step)
        writer.add_scalar("Dice/test", avg_dice, total_test_step)
        writer.add_scalar("IOU/test", avg_iou, total_test_step)

        # 更新学习率（用 Dice 做指标）
        scheduler.step(avg_dice)

        # ============= EarlyStopping + 保存模型 =============
        if avg_dice > best_dice + 1e-6:
            best_dice = avg_dice
            no_improve = 0
            torch.save(net.state_dict(), "model/best_model.pth")
            print(f"保存最佳模型：epoch={i}, Dice={best_dice:.4f}")
        else:
            no_improve += 1
            print(f"验证集 Dice 没提升 ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"早停触发：验证集 Dice {patience} 个 epoch 未提升")
            break


if __name__ == '__main__':
    train_model()





