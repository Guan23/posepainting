#!usr/bin/env python
# _*_ encoding: utf-8 _*_
# Author: Guan
# Create Time: 2021/12/15 下午3:46


import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as trs
import time
import pandas as pd


# 第一种用GPU训练的方式：
# 分别找到网络模型、数据（data和label）以及损失函数，然后调用.cuda（）并返回即可
# 其实也就是将这三者转移到cuda上进行运算

class MyNet(nn.Module):
    # 定义网络结构
    def __init__(self):
        super(MyNet, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(8, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 3)
        )

    # 前向传播网络
    def forward(self, x):
        output = self.model1(x)
        return output


if __name__ == '__main__':
    print('\n-----------------------------------------START---------------------------------------\n')

    # print(torch.__version__)
    # print(torch.cuda.is_available())  # 查看gpu是否可用
    # print(torch.cuda.device_count())  # 查看gpu数量
    # print(torch.cuda.current_device())  # 查看当前gpu号
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 查看设备名

    data_dir = "../human_size/test.csv"
    data = pd.read_csv(data_dir)
    print(data)

    # 定义训练的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 当然更严谨的写法是这样，使用三目运算符来适应不同设备

    # 1、创建训练集和测试集
    train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=trs.ToTensor(), download=True)
    # test_data = torchvision.datasets.CIFAR10('./dataset', train=False, transform=trs.ToTensor(), download=True)
    # # 2、打印训练集和测试集的基本属性（长度、图像尺寸等）
    # train_data_len = len(train_data)
    # test_data_len = len(test_data)
    # print('train dataset length:{}'.format(train_data_len))
    # print('test dataset length:{}'.format(test_data_len))
    # # 3、利用Dataloader加载数据集
    # train_dataloader = DataLoader(train_data, batch_size=64)
    # test_dataloader = DataLoader(test_data, batch_size=16)
    # # 4、创建网络模型
    # mynet = MyNet()
    # mynet.to(device)
    # # 5、构建损失函数和优化器
    # loss_func = nn.CrossEntropyLoss()
    # loss_func.to(device)
    # learning_rate = 1e-2
    # optimizer = torch.optim.Adam(mynet.parameters(), lr=learning_rate)
    # # 6、设置训练网络的一些超参数
    # total_train_step = 0  # 记录训练的次数
    # total_test_step = 0  # 记录测试的次数
    # epochs = 0  # 总训练轮次
    #
    # # 7、添加可视化
    # writer = SummaryWriter('../logs_mynet')
    #
    # start_time = time.time()  # 记录当前时间
    # # 8、训练并验证网络
    # for i in range(epochs):
    #     print('----------第 {} 轮训练开始----------'.format(i + 1))
    #     # 训练数据
    #     mynet.train()  # 表明接下来要对网络进行训练，这行其实只对网络中有dropout、bn层有用，但规范起见，最好都写上
    #     for data in train_dataloader:
    #         optimizer.zero_grad()  # 梯度归0
    #         imgs, targets = data
    #         # 把imgs和targets送到device设备上
    #         imgs = imgs.to(device)
    #         targets = targets.to(device)
    #         output = mynet(imgs)
    #         loss = loss_func(output, targets)
    #         loss.backward()  # 反向传播
    #         optimizer.step()  # 优化
    #
    #         total_train_step += 1
    #         # 未防止打印过多无用信息，每100次训练打印一次训练信息
    #         if total_train_step % 100 == 0:
    #             # 打印训练的信息，tensor.item()就是把tensor转成对应的数字，然后round四舍五入保留4位小数
    #             print('训练次数为：{}时，loss为：{}'.format(total_train_step, round(loss.item(), 4)))
    #             writer.add_scalar('train_loss', round(loss.item(), 4), total_train_step / 100)  # 添加折线图
    #     end_time = time.time()  # 记录当前时间
    #     print('cost time:{}'.format(round(end_time - start_time, 4)))  # 每轮次训练消耗的时间
    #
    #     # 每次训练完，都做一次验证，即validation，测试过程不需要反向传播，故不需要梯度信息
    #     total_test_loss = 0  # 测试集上总体的loss
    #     total_test_acc = 0
    #     mynet.eval()  # 跟上面的mynet.train()作用类似，验证和测试的时候加上
    #     with torch.no_grad():
    #         for data in test_dataloader:
    #             imgs, targets = data
    #             # 把imgs和targets送到device设备上
    #             imgs = imgs.to(device)
    #             targets = targets.to(device)
    #             outputs = mynet(imgs)
    #             loss = loss_func(output, targets)
    #             total_test_loss += loss.item()  # 计算每次在test集上总体的loss
    #             tp = (outputs.argmax(1) == targets).sum()  # 计算每次在test上的TP
    #             total_test_acc += tp  # 把tp加起来
    #
    #     total_test_step += 1
    #     print("整体测试集的loss为：{}".format(round(total_test_loss, 4)))
    #     print("整体测试集的acc为：{}".format(round(total_test_acc.item() / test_data_len, 4)))
    #     writer.add_scalar('test_loss', round(total_test_loss, 4), total_test_step)
    #     writer.add_scalar('test_acc', round(total_test_acc.item() / test_data_len, 4), total_test_step)
    #     torch.save(mynet, './saved/mynet_{}.pth'.format(i))  # 把每一轮次的pth模型保存下来
    #
    # writer.close()

    print('\n------------------------------------------END----------------------------------------\n')
