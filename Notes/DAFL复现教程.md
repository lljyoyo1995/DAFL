# DAFL复现教程

[Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)

[源码](https://github.com/andudu/DAFL)

ICCV 2019

# Teacher网络

## 网络结构

```python
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            return output,feature
```

## teacher-train

```text
Train - Epoch 1, Batch: 1, Loss: 2.297301
Test Avg. Loss: 0.000191, Accuracy: 0.970800
Train - Epoch 2, Batch: 1, Loss: 0.083168
Test Avg. Loss: 0.000126, Accuracy: 0.979500
Train - Epoch 3, Batch: 1, Loss: 0.053707
Test Avg. Loss: 0.000100, Accuracy: 0.983700
Train - Epoch 4, Batch: 1, Loss: 0.035690
Test Avg. Loss: 0.000077, Accuracy: 0.988300
Train - Epoch 5, Batch: 1, Loss: 0.010872
Test Avg. Loss: 0.000101, Accuracy: 0.985000
Train - Epoch 6, Batch: 1, Loss: 0.059809
Test Avg. Loss: 0.000080, Accuracy: 0.987800
Train - Epoch 7, Batch: 1, Loss: 0.019303
Test Avg. Loss: 0.000062, Accuracy: 0.990800
Train - Epoch 8, Batch: 1, Loss: 0.002754
Test Avg. Loss: 0.000096, Accuracy: 0.986300
Train - Epoch 9, Batch: 1, Loss: 0.065059
Test Avg. Loss: 0.000080, Accuracy: 0.988500
```

# Student网路

## 网络结构

```python
class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 42)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(42, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 60)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            return output,feature
```

## student-train

```text
[Epoch 0/10] [loss_oh: 0.318259] [loss_ie: -0.842642] [loss_a: -1.169433] [loss_kd: 1.606884]
Test Avg. Loss: 0.078693, Accuracy: 0.088900
[Epoch 1/10] [loss_oh: 0.320466] [loss_ie: -0.961918] [loss_a: -1.096539] [loss_kd: 1.566164]
Test Avg. Loss: 0.072762, Accuracy: 0.098200
[Epoch 2/10] [loss_oh: 0.377669] [loss_ie: -0.973918] [loss_a: -1.077750] [loss_kd: 1.514326]
Test Avg. Loss: 0.072715, Accuracy: 0.159800
[Epoch 3/10] [loss_oh: 0.369518] [loss_ie: -0.984305] [loss_a: -1.051915] [loss_kd: 1.499127]
Test Avg. Loss: 0.074644, Accuracy: 0.135300
[Epoch 4/10] [loss_oh: 0.271418] [loss_ie: -0.975467] [loss_a: -1.100046] [loss_kd: 1.504842]
Test Avg. Loss: 0.065918, Accuracy: 0.305200
[Epoch 5/10] [loss_oh: 0.306910] [loss_ie: -0.987264] [loss_a: -1.261602] [loss_kd: 1.368728]
Test Avg. Loss: 0.060659, Accuracy: 0.288100
[Epoch 6/10] [loss_oh: 0.313782] [loss_ie: -0.983497] [loss_a: -1.283380] [loss_kd: 1.260229]
Test Avg. Loss: 0.058173, Accuracy: 0.343500
[Epoch 7/10] [loss_oh: 0.356809] [loss_ie: -0.988510] [loss_a: -1.226739] [loss_kd: 1.138210]
Test Avg. Loss: 0.052669, Accuracy: 0.459000
[Epoch 8/10] [loss_oh: 0.360702] [loss_ie: -0.980808] [loss_a: -1.176021] [loss_kd: 1.025557]
Test Avg. Loss: 0.058354, Accuracy: 0.363600
[Epoch 9/10] [loss_oh: 0.369194] [loss_ie: -0.984903] [loss_a: -1.138101] [loss_kd: 1.040750]
Test Avg. Loss: 0.054607, Accuracy: 0.402600
```

# Teacher网络与Student网络对比

+ Teacher网络结构与Student网络结构一样，都是LeNet5网络；
+ Student网络相对于Teacher网络，Conv2d卷积核的数量减半，即Conv2d卷积核的输出通道数减半，对应的output输出尺寸减半；
+ 如何确定Student网络结构：
  + 根据用户提供的pb或者ckpt等模型文件，解析Teacher网络。根据Teacher网络结构的特性，设计一种自动化生成Student网络的算法，生成对应的Student网络；
  + 根据不同的任务，比如分类、目标检测、图像分割等，自动匹配对应的Student网络（工具箱提供Model ZOO）；



# 运行环境

## 系统环境

```text
系统：windows10,64位
显卡：NVIDIA GeForce GTX 1050Ti,4GB
内存：16GB
Python版本：3.8.10
```

## 依赖包

```text
certifi==2021.5.30
charset-normalizer==2.0.4
idna==3.2
numpy==1.21.1
Pillow==8.3.1
requests==2.26.0
torch==1.8.0+cu101
torchvision==0.9.0+cu101
typing-extensions==3.10.0.0
urllib3==1.26.6
wincertstore==0.2
```

# 修改源码

## teacher-train.py

```python
# 修改data路径，修改output_dir路径
parser.add_argument('--data', type=str, default='E:/MyDocuments/DAFL/data/')
parser.add_argument('--output_dir', type=str, default='E:/MyDocuments/DAFL/models/')
```

```python
# 在线下载MNIST数据集
data_train = MNIST(args.data, download=True,
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
```

```python
# 修改batch_size大小
data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=4)
# 修改num_workers大小
data_test_loader = DataLoader(data_test, batch_size=512, num_workers=4)
```

## DAFL-train.py

```python
# 修改data路径
parser.add_argument('--data', type=str, default='E:/MyDocuments/DAFL/data/')
# 修改teacher_dir路径
parser.add_argument('--teacher_dir', type=str, default='E:/MyDocuments/DAFL/models/')
# 修改batch_size大小
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
# 修改output_dir路径
parser.add_argument('--output_dir', type=str, default='E:/MyDocuments/DAFL/models/')
```

```python
# 修改batch_size大小
data_test_loader = DataLoader(data_test, batch_size=32, num_workers=1, shuffle=False)  # test阶段需要使用有标注的数据
```



# 可能存在的问题

+ RuntimeError:
          An attempt has been made to start a new process before the
          current process has finished its bootstrapping phase.

  This probably means that you are not using fork to start your
      child processes and you have forgotten to use the proper idiom
      in the main module:

  if __name__ == '__main__':
          freeze_support()
          ...

  The "freeze_support()" line can be omitted if the program
  is not going to be frozen to produce an executable.

  **参考资料**

  [pyTorch：The "freeze_support()" line can be omitted if the program is not going to be frozen](https://blog.csdn.net/zzq060143/article/details/87863354)

```text
错误原因：
没有main主函数 if __name__ == '__main__':

解决办法：
将程序放入 if __name__ == '__main__':
```

+ 显存不足，导致程序运行中止

```python
try:
    outputs_T, features_T = teacher(gen_imgs, out_feature=True)  # Teacher网络也不需要标注数据
except RuntimeError as exception:
    if 'out of memory' in str(exception):
        print('WARNING out of memory')
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
```

