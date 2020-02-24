## 卷积神经网络基础

从CNN的基本网络构成讲起：**卷积-池化-全连接**

### 二维卷积层

二维互相关（cross-correlation）运算的`输入`是一个二维输入数组和一个二维核（kernel）数组，`输出`也是一个二维数组，其中**核数组**通常称为`卷积核`或**过滤器（filter）**。

【运算过程】卷积核的尺寸通常小于输入数组，卷积核在输入数组上滑动，在每个位置上，卷积核与该位置处的输入子数组**按元素相乘并求和**，得到输出数组中相应位置的元素。

【模型参数】二维卷积层将`输入`和`卷积核`做互相关运算，并加上一个`标量偏置`来得到输出。卷积层的模型参数包括**卷积核**和**标量偏置**。

注意：卷积层得名于卷积运算，但**卷积层中用到**的并非卷积运算而**是互相关运算**。我们将核数组上下翻转、左右翻转，再与输入数组做互相关运算，这一过程就是卷积运算。

**二维卷积层`输出`的二维数组**可以看作是输入在空间维度（**宽和高**）上某一级的**表征**，也叫`特征图（feature map）`

**影响元素的前向计算的所有可能`输入`区域**（可能大于输入的实际尺寸）叫做`感受野（receptive field）`

### 填充和步幅

`填充（padding）`是指在输入高和宽的两侧填充元素（通常是0元素）。
在互相关运算中，卷积核在输入数组上滑动，每次滑动的行数与列数即是`步幅（stride）`。

### 1x1卷积层

- 1x1卷积核可在不改变高宽的情况下，调整通道数。
- 1x1卷积核不识别高和宽维度上相邻元素构成的模式，其主要计算发生在通道维上。
- 假设我们将通道维当作**特征维**，将高和宽维度上的元素当成**数据样本**，那么**1x1卷积层的作用与全连接层等价**。

### 卷积层与全连接层的对比

二维卷积层经常用于处理图像，与此前的全连接层相比，它主要有两个优势：

> 一是全连接层把图像**展平**成一个向量，在输入图像上相邻的元素可能因为展平操作不再相邻，网络难以捕捉局部信息。而卷积层的设计，天然地具有**提取局部信息的能力**。
> 二是**卷积层的参数量更少**。使用卷积层可以以较少的参数数量来处理更大的图像。

### 卷积层的简洁实现

我们使用Pytorch中的`nn.Conv2d`类来实现二维卷积层，主要关注以下几个构造函数参数：

- `in_channels` (python:int) – Number of channels in the input imag
- `out_channels` (python:int) – Number of channels produced by the convolution
- `kernel_size` (python:int or tuple) – Size of the convolving kernel
- `stride` (python:int or tuple, optional) – Stride of the convolution. Default: 1
- `padding` (python:int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
- `bias` (bool, optional) – If True, adds a learnable bias to the output. Default: True

```python
conv2d = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 5), stride=1, padding=(1, 2))
```

### 池化

- 池化层主要用于缓解卷积层对位置的过度敏感性。
  同卷积层一样，池化层每次对输入数据的一个固定形状窗口（又称池化窗口）中的元素计算输出
  【运算过程】池化层直接计算池化窗口内元素的最大值或者平均值，该运算也分别叫做最大池化或平均池化。
- 池化层也可以在输入的高和宽两侧**填充**并调整窗口的移动**步幅**来改变输出形状。
  池化层填充和步幅与卷积层填充和步幅的工作机制一样。
  在处理多通道输入数据时，池化层对**每个输入通道**分别池化，但不会像卷积层那样将各通道的结果按通道相加。这意味着**池化层的输出通道数与输入通道数相等**。

### 池化层的简洁实现

我们使用Pytorch中的`nn.MaxPool2d`实现最大池化层，关注以下构造函数参数：

- `kernel_size` – the size of the window to take a max over
- `stride` – the stride of the window. Default value is kernel_size
- `padding` – implicit zero padding to be added on both sides

```python
pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
```

## LeNet

### 简介

![Image Name](https://cdn.kesci.com/upload/image/q5ndwsmsao.png?imageView2/0/w/960/h/960)

LeNet分为`卷积层块`和`全连接层块`两个部分。`卷积层块`里的基本单位是**卷积层**后接**平均池化层**：卷积层用来识别图像里的空间模式，如线条和物体局部，之后的平均池化层则用来降低卷积层对位置的敏感性(平移不变性)。

使用**全连接层的`局限性`**：

- 图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别。
- 对于大尺寸的输入图像，使用全连接层容易导致模型过大。

使用**卷积层的`优势`**：

- 卷积层保留输入形状。
- 卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

### LeNet模型结构与代码实现

![Image Name](https://cdn.kesci.com/upload/image/q5ndxi6jl5.png?imageView2/0/w/640/h/640)

```python
net = torch.nn.Sequential(     #LeNet
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28
    nn.Sigmoid(),                                                       
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5
    Flatten(),                                                          #b*16*5*5   => b*400
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
```

### 相关pytorch函数

> (1). `net.train()`
> 启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True
> (2). `net.eval()`
> 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False

## 卷积神经网络进阶

### 深度卷积神经网络（AlexNet）

`LeNet`: 在大的真实数据集上的表现并不尽如⼈意。存在问题：

- 神经网络计算复杂。
- 还没有大量深⼊研究参数初始化和非凸优化算法等诸多领域。

`AlexNet`首次证明了学习到的特征可以超越手工设计的特征，从而一举打破计算机视觉研究的前状。特征如下：

- 8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
- 将Sigmoid激活函数改成了更加简单的ReLU激活函数。
- 用Dropout来控制全连接层的模型复杂度。
- 引入数据增强，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。

![Image Name](https://cdn.kesci.com/upload/image/q5kv4gpx88.png?imageView2/0/w/640/h/640)

```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            #由于使用CPU镜像，精简网络，若为GPU镜像可添加该层
            #nn.Linear(4096, 4096),
            #nn.ReLU(),
            #nn.Dropout(0.5),

            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )
```

### 使用重复元素的网络（VGG）

VGG：通过重复使⽤简单的基础块来构建深度模型。
Block:数个相同的填充为1、窗口形状3x3为的卷积层,接上一个步幅为2、窗口形状2x2为的最大池化层。
卷积层保持输入的高和宽不变，而池化层则对其减半。

![Image Name](https://cdn.kesci.com/upload/image/q5l6vut7h1.png?imageView2/0/w/640/h/640)

```python
def vgg_block(num_convs, in_channels, out_channels): #卷积层个数，输入通道数，输出通道数
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net
```

### 网络中的网络（NiN）

LeNet、AlexNet和VGG：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。
NiN：串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络。
用了输出通道数等于标签类别数的NiN块，然后使用全局平均池化层对每个通道中所有元素求平均并直接用于分类。

1×1卷积核作用
1.放缩通道数：通过控制卷积核的数量达到通道数的放缩。
2.增加非线性。1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性。
3.计算参数少

![Image Name](https://cdn.kesci.com/upload/image/q5l6u1p5vy.png?imageView2/0/w/960/h/960)
NiN重复使用由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络。
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数 的NiN块和全局平均池化层。
NiN的以上设计思想影响了后面一系列卷积神经网络的设计。

```python
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk

class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(), 
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    d2l.FlattenLayer())
```

### GoogLeNet

1. 由Inception基础块组成。
2. Inception块相当于一个有4条线路的子网络。它通过不同窗口形状的卷积层和最大池化层来并行抽取信息，并使⽤1×1卷积层减少通道数从而降低模型复杂度。
3. 可以自定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度。

![Image Name](https://cdn.kesci.com/upload/image/q5l6uortw.png?imageView2/0/w/640/h/640)

```python
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
```

完整模型结构

![Image Name](https://cdn.kesci.com/upload/image/q5l6x0fyyn.png?imageView2/0/w/640/h/640)

```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   d2l.GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5, 
                    d2l.FlattenLayer(), nn.Linear(1024, 10))

net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))
```
