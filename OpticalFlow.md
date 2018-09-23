《FlowNet: Learning Optical Flow with Convolutional Networks》

(1)FlowNetSimple：

把两张图片stack到一起,在传入一个CNN里，输出optical flow；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/40.png)

(2)FlowNetCorr：

两张图片分别提取特,在某一层feature maps上计算相关性,

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/41.png)

为了减小计算量，在某一个位置上，只对位移大小为d的邻域计算相关性,即邻域大小D:=2d+1，

Refinement的网络结构为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/42.png)

把network收缩部分的特征和deconv得到的特征以及前一个flow map(如果存在的话)upsample得到的特征concatenate到一起，进行deconv得到下一层的特征，进行conv得到本层的flow map。

《FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks》

stack 《FlowNet》里的网络结构：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/43.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/44.png)
