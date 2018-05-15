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

（1）《Detecting Oriented Text in Natural Images by Linking Segments》

待补充

（2）《Deep Direct Regression for Multi-Oriented Scene Text Detection》

待补充

（3）《Fused Text Segmentation Networks for Multi-oriented Scene Text Detection》

待补充

（4）《IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection》

待补充

