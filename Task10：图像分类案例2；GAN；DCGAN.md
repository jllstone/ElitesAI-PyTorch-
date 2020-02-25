## 图像分类案例2

### finetune（微调）

```python
def get_net(device):
    finetune_net = models.resnet34(pretrained=False)  # 预训练的resnet34网络
    finetune_net.load_state_dict(torch.load('/home/kesci/input/resnet347742/resnet34-333f7ec4.pth'))
    for param in finetune_net.parameters():  # 冻结参数
        param.requires_grad = False
    # 原finetune_net.fc是一个输入单元数为512，输出单元数为1000的全连接层
    # 替换掉原finetune_net.fc，新finetuen_net.fc中的模型参数会记录梯度
    finetune_net.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=120)  # 120是输出类别数
    )
    return finetune_net
```

### GAN

GAN的主要灵感来源于博弈论中零和博弈的思想，应用到深度学习神经网络上来说，就是通过生成网络G（Generator）和判别网络D（Discriminator）不断博弈，进而使G学习到数据的分布，如果用到图片生成上，则训练完成后，G可以从一段随机数中生成逼真的图像。G， D的主要功能是：

- G是一个生成式的网络，它接收一个随机的噪声z（随机数），通过这个噪声生成图像

- D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片

  生成对抗网络的目标是其模型可生成符合数据集分布，又和原数据集不同的数据，并且生成器生成的数据能够骗过分类器。

The generator generates the image as much closer to the true image as possible to fool the discriminator, via maximizing the cross-entropy loss, i.e., maxlog(D(x′)) .

The discriminator tries to distinguish the generated images from the true images, via minimizing the cross-entropy loss, i.e., min−ylogD(x)−(1−y)log(1−D(x)) .

### DCGAN

DCGAN相对于原始的GAN并没有太大的改进，只是将全卷积神经网络应用到了GAN中，因此GAN存在的许多问题DCGAN依然有。

细节方面，DCGAN做了如下改进：

> 取消pooling层。G中用反卷积进行上采样，D中用加入stride的卷积代替pooling
> batch normalization
> 去掉FC层，网络为全卷积网络
> G中使用Relu(最后一层用tanh)
> D中用LeakyRelu
