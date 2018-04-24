2018.04.18

卷积神经网络的解释性

判断图片像素值对最终分类结果的影响的两种方法：

（1）《Learning Deep Features for Discriminative Localization》
生成特征权重图CAM（Class Activation Mapping）

（2）《Methods for Interpreting and Understanding Deep Neural Networks》
最终输出对图片像素值的导数（的平方）


2018.04.24

文字检测

（1）《Multi-Oriented Text Detection with Fully Convolutional Networks》

先用FCN 得到候选的文字区域，再用传统方法分割出一行行文字和文字方向，再用FCN得到每个字符的中心，再进一步分类文字和非文字；

（2）《Scene Text Detection via Holistic, Multi-Channel Prediction》

使用FCN生成三张map，一张分割行文字，一张分割字符，一张回归每个像素点对应的文字方向；

（3）《Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection》

使用CNN+带方向的滑动窗口来回归文字区域，使用Monte-Carlo方法来计算不规则四边形的重合面积；

（4）《EAST: An Efficient and Accurate Scene Text Detector》

使用FCN回归每个像素点是文字的score，以及对应的框的位置，再把框合并。
