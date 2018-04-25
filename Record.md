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

                                                                                                                                          
2018.04.25

行文字识别

（1）最简单的思路：滑动窗口，每个窗口使用CNN分类，把得到的结果序列处理一下得到最终识别结果。缺点：处理分类结果序列时，何时该合并相邻的同样的分类结果，何时不合并很难判断；

（2）滑动窗口，每个窗口使用CNN分类，把得到的结果序列使用CTC来得到最终结果，解决（1）的问题。缺点：每个窗口只有窗口内的像素值信息，缺少上下文联系；

（3）滑动窗口，每个窗口使用CNN分类，把得到的结果序列再传入RNN（比如两层的双向LSTM），再把RNN的输出结果使用CTC合并。缺点：滑动窗口会有重叠，重叠部分要做同样的卷积计算两次；（《Reading Scene Text in Deep Convolutional Sequences》）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/1.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/2.png)

（4）把整行图片传入CNN，得到高固定、宽任意、通道数固定的特征，以宽度作为时间生成序列，传入RNN，再把RNN的输出结果使用CTC合并，也就是所谓的“CRNN”；（《An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition》）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/3.png)

（5）把整行图片传入CNN，得到高固定、宽任意、通道数固定的特征，以宽度作为时间生成序列，传入RNN，在RNN中使用Attention机制代替CTC，得到最终识别结果。（《Recursive Recurrent Nets with Attention Modeling for OCR in the Wild》，《Robust Scene Text Recognition with Automatic Rectification》）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/4.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/5.png)

