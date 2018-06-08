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

先用FCN得到候选的文字区域，再用传统方法分割出一行行文字和文字方向，再用FCN得到每个字符的中心，再进一步分类文字和非文字；

（2）《Scene Text Detection via Holistic, Multi-Channel Prediction》

使用FCN生成三张map，一张分割行文字，一张分割字符，一张回归每个像素点对应的文字方向；

（3）《Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection》

使用CNN+带方向的滑动窗口来回归文字区域，使用Monte-Carlo方法来计算不规则四边形的重合面积；

（4）《EAST: An Efficient and Accurate Scene Text Detector》

使用FCN回归每个像素点是文字的score，以及对应的框的位置，再把框合并，对较长文字的检测效果不太好，两端会有漏掉的部分，可能时因为网络的Receptive Field太小。

                                                                                                                                         

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

补充：另一篇使用attention机制的文章是《Attention-based Extraction of Structured Information from Street View Imagery》，与（5）中两篇文章的不同之处是：（5）中文章使用的网络结构是 CNN-RNN(encoder) + Attention-RNN(decoder)，而这篇文章的网络结构是 CNN(encoder) + Attention-RNN(decoder)。原因是：（5）中的文章是检测一行文字，图片长度是变化的，需要先使用RNN转成固定长度的特征，而这篇文章检测的路牌是多行文字，图片尺寸固定，所以可以直接把最后一个卷积层作为特征。



2018.05.10

word2vector

（1）《A Neural Probabilistic Language Model》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/6.png)

训练的文本作为输入，在文本上用back-off n-gram模型得到的概率作为输出，训练模型参数，g可以是前向神经网络或者循环神经网络，C是学到的distributed representation；

（2）《Efficient Estimation of Word Representations in Vector Space》、《Exploiting Similarities among Languages for Machine Translation》、《Distributed Representations of Words and Phrases and their Compositionality》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/7.png)

两个模型都各有一个input representation u和一个output representation v，w(t)是1-of-V coding（V是字典里words的个数），

Continuous Bag-of-Words (CBOW)

计算窗口内的所有words（除了当前位置i）的u的和的平均值，再与每个v做内积得到对应每个word的输出，再用softmax计算概率；

Skip-gram

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/8.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/9.png)

两种方法都不考虑窗口内words的顺序，都是为了训练得到words的向量表示。

（3）《Linguistic Regularities in Continuous Space Word Representations》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/10.png)

使用RNN结构，隐藏状态保存句子历史信息，u是学到的word representations。

（4）《Linguistic Regularities in Sparse and Explicit Word Representations》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/11.png)

分别使用explicit vector representations（见上图）和neural embeddings两种words representation方法，分别用3COSADD和PAIRDIRECTION作为优化目标函数，训练模型。

3COSADD

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/12.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/13.png)

PAIRDIRECTION

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/14.png)

（5）《Learning word embeddings efficiently with noise-contrastive estimation》（vLBL and ivLBL）

待补充

（6）《GloVe: Global Vectors for Word Representation》 （GloVe）

待补充

2018.05.14

text classification

（1）最简单的方法：对文本中的每个word的representation vector做加权平均，得到的vector作为文本的vector；

（2）《Distributed Representations of Sentences and Documents》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/15.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/16.png)

监督学习，分为两步：1）在训练集上训练word vectors W和paragraph vectors D，2）在推断阶段，固定W，训练D，得到测试集文本的paragraph vectors，再使用分类器以D为特征训练分类模型；

（3）《Deep Unordered Composition Rivals Syntactic Methods for Text Classification》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/17.png)

监督学习，使用文本里words的vectors的均值，使用更深的feed-forward network，及使用dropout（训练时随机漏掉文本中的某些word）。



2018.05.15

文字检测（续一）

（1）《Arbitrary-Oriented Scene Text Detection via Rotation Proposals》

待补充

（2）《Detecting Oriented Text in Natural Images by Linking Segments》

待补充

（3）《Deep Direct Regression for Multi-Oriented Scene Text Detection》

待补充

（4）《Fused Text Segmentation Networks for Multi-oriented Scene Text Detection》

待补充

（5）《IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection》

待补充

2018.05.17

RCNN演进史

背景：CNN最早提出作为一个图片的分类算法，得到比之前高很多的准确度。用分类的方法检测：滑动窗口，在图片上滑动，每个位置做一次分类，再把所有分类结果整合在一起。缺点是：窗口的尺寸和长宽比变化太多，所以要进行很多次分类操作，如果使用CNN来分类，非常耗时。

R-CNN：用selective search方法先选出一部分可能存在物体的候选区域，再对每个区域用CNN提取特征，最后用SVM做分类。

优点：减少了要做特征提取的区域数量，

缺点：区域数量还是很多（10的2次方~10的3次方数量级），每个区域做一次CNN，耗时。

Fast RCNN：对图片整体做卷积操作，再把选出的区域对应到最后一层卷积层获取卷积特征，利用SPPnet的方法，把不同尺寸、长宽比的特征归一化为同样大小，再用softmax进行分类。

R-CNN之所以要对每一个区域做一次CNN，是因为CNN里的fc是固定大小，所以输入的图片尺寸大小要一致，而SPPnet则解决了这个问题。SPPnet放在fc之前、卷积层之后，把不同大小的卷积特征归一化为同样大小。（卷积操作不要求图片大小固定）

Faster RCNN：Fast RCNN最耗时的操作为selective search，于是提出RPN，把候选区域的提取用CNN来完成，实现端到端的训练。

注：Fast RCNN中的ROIpool和SPPnet的不同：SPPnet pool多个尺度来整合成同样大小，但失去了对应的空间位置；ROIpool把特征map分成H * W（H、W为归一化后的宽高）个subwindow，每个subwindow里做pool。


2018.06.04

SFace脉络：

《Multi-view Face Detection Using Deep Convolutional Neural Networks》

使用修改过的AlexNet，全卷积网络，每个octave缩放3次，最后一层生成1张heatmap，根据门限值判断哪些位置是人脸，可以检测不同角度的人脸，但无法给出具体角度。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/18.png)

《DenseBox: Unifying Landmark Localization with End to End Object Detection》

全卷积网络，输出heatmap和左上\右下两个点的相对位置，类似于EAST算法，使用多个尺寸测试，可以检测关键点。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/19.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/20.png)

《UnitBox: An Advanced Object Detection Network》

使用DenseBox类似的结构，把box回归的L2 loss改成IOU loss。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/21.png)

《SFace: An Efficient Network for Face Detection in Large Scale Variations》

整合anchor-based和anchor-free两种检测方法，都使用IOU loss，再把两种检测方法的结果融合。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/22.png)

注：《Convolutional Face Finder: A Neural Architecture for Fast and Robust Face Detection》一篇2004年发表的论文，提出用CNN检测人脸。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/23.png)


2018.06.06

Highway Network 和 ResNet

《Deep Residual Learning for Image Recognition》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/24.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/25.png)

《Training Very Deep Networks》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/26.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/27.png)


2018.06.08

Style Transfer

《A Neural Algorithm of Artistic Style》、《Image Style Transfer Using Convolutional Neural Networks》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/28.png)

Content Representation Loss:

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/29.png)

Gram matrix:

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/30.png)

Style Representation Loss:

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/31.png)

Total Loss:

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/32.png)

《preserving color in neural artisic style transfer》

保持颜色的风格变换的两种方法：

1、Color histogram matching

变换style image的颜色，使变换后的style image的颜色的均值和协方差与content image相等，文章中使用线性变换，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/33.png)

具体使用的是Image Analogies color transfer和Cholesky transfer，《Colour mapping: A review of recent methods, extensions and applications》中有更多变换方法。

2、Luminance-only transfer

使用YIQ颜色空间，只在亮度Y通道上做风格变换，如果style image和content image的亮度通道的直方图不一致，可以先修改style image的亮度通道使得一致，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/34.png)

