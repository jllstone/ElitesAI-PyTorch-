## 线性回归

掌握机器学习，从线性回归开始

### 线性回归的基本要素

- 数据集
- 模型——线性回归
- 损失函数——均方误差
- 优化函数——随机梯度下降
  - 优化函数的有以下两个步骤：
    - (i)初始化模型参数，一般来说使用随机初始化
    - (ii)在数据上迭代多次，通过在负梯度方向移动参数来更新每个参数

### PyTorch中的相关函数

> torch.Tensor.view()
> 作用: 将输入的torch.Tensor改变形状(size)并返回,view起到的作用是reshape
>
> 初始化参数方法
> torch.nn.init.normal*(net[0].weight, mean=0.0, std=0.01)*
>
> torch.nn.init.constant(net[0].bias, val=0.0)
>
> 定义均方误差
> torch.nn.MSELoss()
>
> 定义sgd优化方法
> torch.optim.SGD(net.parameters(), lr=0.03)
>
> optimizer.step()通常用在每个mini-batch之中，只有用了optimizer.step()，模型才会更新
>
> torch.randn(sizes, out=None)
> 返回一个张量，包含了从标准正态分布（均值为0，方差为1）中抽取的一组随机数。
>
> backward()是PyTorch中提供的函数，用于求梯度，配套有require_grad：
> 所有的tensor都有requires_grad属性,可以设置这个属性:x = 
>
> tensor.ones(2,4,requires_grad=True)
> 如果想改变这个属性，调用tensor.requires*grad*()方法：x.requires*grad*(False)
>
> torch.mm 和 torch.mul 的区别？
>
> torch.mm是矩阵相乘，torch.mul是按元素相乘
> torch.manual_seed(1)的作用？
>
> 设置随机种子，使实验结果可以复现
> optimizer.zero_grad()的作用？
>
> 清零梯度，防止不同batch得到的梯度累加

### 建模的一般步骤

1.  定义模型
2. 初始化模型参数
3. 定义损失函数
4. 定义优化函数
5. 模型训练

### 三种堆叠模型的方法

```python
# method one
net = nn.Sequential(    
    nn.Linear(num_inputs, 1)    
    # other layers can be added here
	)

# method two
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([          
    ('linear', nn.Linear(num_inputs, 1))          
    # ......        
	]))
```

## Softmax与分类模型

### softmax回归

**softmax回归**实际上解决的是一个“**分类问题**”

**softmax回归**同**线性回归**一样，也是一个**单层神经网络**

softmax回归的**输出层**也是一个**全连接层**

- 输出问题

  直接使用输出层的输出有两个问题：

  1. 一方面，由于输出层的输出值的范围不确定，我们难以直观上判断这些值的意义。
  2. 另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。

  softmax运算符（softmax operator）解决了以上两个问题。

  即`通过将输出值变换成值为正且和为1的概率分布`，解决了真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量的问题。

### 交叉熵损失函数

想要预测分类结果正确，我们其实并不需要预测概率完全等于标签概率

平方损失则过于严格

改善上述问题的一个方法是使用更适合衡量两个概率分布差异的测量函数(这里考虑二分类问题)

其中，交叉熵（cross entropy）是一个常用的衡量方法

交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。

交叉熵损失函数定义为：

最小化交叉熵损失函数 等价于 最大化训练数据集所有标签类别的联合预测概率。

### PyTorch中的相关函数

多维Tensor按维度操作

```python
X.sum(dim=0, keepdim=True)  # dim为0，按照相同的列求和，并在结果中保留列特征
X.sum(dim=1, keepdim=True)  # dim为1，按照相同的行求和，并在结果中保留行特征
X.sum(dim=0, keepdim=False) # dim为0，按照相同的列求和，不在结果中保留列特征
X.sum(dim=1, keepdim=False) # dim为1，按照相同的行求和，不在结果中保留行特征

tensor([[5, 7, 9]])
tensor([[ 6],        
        [15]])
tensor([5, 7, 9])
tensor([ 6, 15])
```



> torch.gather(input, dim, index, out=None)torch.gather的作用是这样的，index实际上是索引，具体是行还是列的索引要看前面dim 的指定，指定dim=1，也就是横向，那么索引就是列号，index的大小就是输出的大小。
>
> 定义交叉熵损失
> nn.CrossEntropyLoss()

### 训练时训练集和测试集初始误差关系

训练数据集上的准确率`低于`测试数据集上的准确率，原因是：

训练集上的准确率是在一个epoch的过程中计算得到的，

测试集上的准确率是在一个epoch结束后计算得到的，后者的模型参数更优

## 多层感知机

### 多层感知机的基本知识

深度学习主要关注多层模型。

在这里，我们将以多层感知机（multilayer perceptron，MLP）为例，介绍多层神经网络的概念。

多层感知机就是含有至少一个**隐藏层**的由全连接层组成的神经网络，且每个隐藏层的输出通过**激活函数**进行变换。多层感知机的`层数`和各隐藏层中`隐藏单元个数`都是**超参数**。

**隐藏层**

![Image Name](https://cdn.kesci.com/upload/image/q5ho684jmh.png)

**激活函数**

添加隐藏层，依然只能与仅含输出层的单层神经网络等价，问题的根源在于全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换的叠加仍然是一个仿射变换。解决问题的一个方法是引入非线性变换，这个非线性函数被称为激活函数（activation function）。

- 常用的激活函数：

1. ReLU函数
   ReLU（rectified linear unit）函数提供了一个很简单的非线性变换，ReLU函数只保留正数元素，并将负数元素清零
2. Sigmoid函数
   sigmoid函数可以将元素的值变换到0和1之间
3. tanh函数
   tanh（双曲正切）函数可以将元素的值变换到-1和1之间：虽然该函数的形状和sigmoid函数的形状很像，但tanh函数在坐标系的原点上对称

- 激活函数的选择

1. ReLu函数是一个通用的激活函数，目前在大多数情况下使用，ReLU函数只能在隐藏层中使用。
2. 用于分类器时，sigmoid函数及其组合通常效果更好。由于梯度消失问题，有时要避免使用sigmoid和tanh函数。
3. 在神经网络层数较多的时候，最好使用ReLu函数，ReLu函数比较简单计算量少，而sigmoid和tanh函数计算量大很多。
4. 在选择激活函数的时候可以先选用ReLu函数如果效果不理想可以尝试其他激活函数。

### PyTorch中的相关函数

> 返回两者较大值
> torch.max(input=X, other=torch.tensor(0.0))
>
> 定义relu函数
> nn.ReLU()
>
> tensor.detach()
> 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
