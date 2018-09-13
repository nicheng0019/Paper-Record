2018.04.18

卷积神经网络的解释性

判断图片像素值对最终分类结果的影响的两种方法：

1、《Learning Deep Features for Discriminative Localization》
生成特征权重图CAM（Class Activation Mapping）

2、《Methods for Interpreting and Understanding Deep Neural Networks》
最终输出对图片像素值的导数（的平方）

                                                                                                                                         

2018.04.24

文字检测

《Multi-Oriented Text Detection with Fully Convolutional Networks》

先用FCN得到候选的文字区域，再用传统方法分割出一行行文字和文字方向，再用FCN得到每个字符的中心，再进一步分类文字和非文字；

《Scene Text Detection via Holistic, Multi-Channel Prediction》

使用FCN生成三张map，一张分割行文字，一张分割字符，一张回归每个像素点对应的文字方向；

《Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection》

使用CNN+带方向的滑动窗口来回归文字区域，使用Monte-Carlo方法来计算不规则四边形的重合面积；

《EAST: An Efficient and Accurate Scene Text Detector》

使用FCN回归每个像素点是文字的score，以及对应的框的位置，再把框合并，对较长文字的检测效果不太好，两端会有漏掉的部分，可能时因为网络的Receptive Field太小。

                                                                                                                                         

2018.04.25

行文字识别

1、最简单的思路：滑动窗口，每个窗口使用CNN分类，把得到的结果序列处理一下得到最终识别结果。缺点：处理分类结果序列时，何时该合并相邻的同样的分类结果，何时不合并很难判断；

2、滑动窗口，每个窗口使用CNN分类，把得到的结果序列使用CTC来得到最终结果，解决(1)的问题。缺点：每个窗口只有窗口内的像素值信息，缺少上下文联系；

3、滑动窗口，每个窗口使用CNN分类，把得到的结果序列再传入RNN（比如两层的双向LSTM），再把RNN的输出结果使用CTC合并。缺点：滑动窗口会有重叠，重叠部分要做同样的卷积计算两次；（《Reading Scene Text in Deep Convolutional Sequences》）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/1.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/2.png)

4、把整行图片传入CNN，得到高固定、宽任意、通道数固定的特征，以宽度作为时间生成序列，传入RNN，再把RNN的输出结果使用CTC合并，也就是所谓的“CRNN”；（《An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition》）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/3.png)

5、把整行图片传入CNN，得到高固定、宽任意、通道数固定的特征，以宽度作为时间生成序列，传入RNN，在RNN中使用Attention机制代替CTC，得到最终识别结果。（《Recursive Recurrent Nets with Attention Modeling for OCR in the Wild》，《Robust Scene Text Recognition with Automatic Rectification》）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/4.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/5.png)

补充：另一篇使用attention机制的文章是《Attention-based Extraction of Structured Information from Street View Imagery》，与(5)中两篇文章的不同之处是：(5)中文章使用的网络结构是 CNN-RNN(encoder) + Attention-RNN(decoder)，而这篇文章的网络结构是 CNN(encoder) + Attention-RNN(decoder)。原因是：(5)中的文章是检测一行文字，图片长度是变化的，需要先使用RNN转成固定长度的特征，而这篇文章检测的路牌是多行文字，图片尺寸固定，所以可以直接把最后一个卷积层作为特征。



2018.05.10

word2vector

《A Neural Probabilistic Language Model》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/6.png)

训练的文本作为输入，在文本上用back-off n-gram模型得到的概率作为输出，训练模型参数，g可以是前向神经网络或者循环神经网络，C是学到的distributed representation；

《Efficient Estimation of Word Representations in Vector Space》、《Exploiting Similarities among Languages for Machine Translation》、《Distributed Representations of Words and Phrases and their Compositionality》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/7.png)

两个模型都各有一个input representation u和一个output representation v，w(t)是1-of-V coding（V是字典里words的个数），

Continuous Bag-of-Words (CBOW)

计算窗口内的所有words（除了当前位置i）的u的和的平均值，再与每个v做内积得到对应每个word的输出，再用softmax计算概率；

Skip-gram

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/8.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/9.png)

两种方法都不考虑窗口内words的顺序，都是为了训练得到words的向量表示。

《Linguistic Regularities in Continuous Space Word Representations》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/10.png)

使用RNN结构，隐藏状态保存句子历史信息，u是学到的word representations。

《Linguistic Regularities in Sparse and Explicit Word Representations》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/11.png)

分别使用explicit vector representations（见上图）和neural embeddings两种words representation方法，分别用3COSADD和PAIRDIRECTION作为优化目标函数，训练模型。

3COSADD

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/12.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/13.png)

PAIRDIRECTION

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/14.png)

《Learning word embeddings efficiently with noise-contrastive estimation》（vLBL and ivLBL）

待补充

《GloVe: Global Vectors for Word Representation》 （GloVe）

待补充



2018.05.14

text classification

1、最简单的方法：对文本中的每个word的representation vector做加权平均，得到的vector作为文本的vector；

2、《Distributed Representations of Sentences and Documents》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/15.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/16.png)

监督学习，分为两步：1）在训练集上训练word vectors W和paragraph vectors D，2）在推断阶段，固定W，训练D，得到测试集文本的paragraph vectors，再使用分类器以D为特征训练分类模型；

3、《Deep Unordered Composition Rivals Syntactic Methods for Text Classification》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/17.png)

监督学习，使用文本里words的vectors的均值，使用更深的feed-forward network，及使用dropout（训练时随机漏掉文本中的某些word）。



2018.05.15

文字检测（续一）

《Arbitrary-Oriented Scene Text Detection via Rotation Proposals》

待补充

《Detecting Oriented Text in Natural Images by Linking Segments》

待补充

《Deep Direct Regression for Multi-Oriented Scene Text Detection》

待补充

《Fused Text Segmentation Networks for Multi-oriented Scene Text Detection》

待补充

《IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection》

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

使用修改过的AlexNet，把FC层改为卷积层，图片大小不限制，图片最小尺寸大于227，最大放大5倍，每个octave缩放3次（octave：八度音，音乐上相邻的八度音的频率比为1：2，所以图像上相邻的octave的尺寸比为1：2），最后一层生成1张heatmap，AlexNet的输入图片的尺寸为227x227，所以heatmap上一个点对应原图像上一个227x227的区域，stride为32，根据heatmap上大于门限值的点得到对应的原图片上区域为人脸，再使用NMS方法过滤得到的区域，可以检测不同角度的人脸，但无法给出具体角度,

两种NMS方法：(1)NMS-max，即传统的NMS，两个boxes的IOU大于门限值，则删掉score小的box，(2)NMS-avg，根据IOU把boxes聚类，每个cluster里把score低的boxes删掉，取剩余的boxes的均值作为最终box的值，cluster里最大的score作为最终的score，EAST里使用的合并文字检测框的方法和这个比较像。

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


《Universal Style Transfer via Feature Transforms》

之前的文章用Gram矩阵来表达风格的特征，这篇文章用signal whitening and coloring transforms (WCTs)来处理content features和style features，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/35.png)

(1)训练Reconstruction decoder：

使用VGG-19来encode特征，再训练与VGG-19对称的decoder来重建图片，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/36.png)

(2)用encoder提取content image和style image的特征：

Whitening transform: ccontent feature先减去均值,再做线性变换，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/37.png)

Coloring transform：style feature先减去均值，再做coloring transform，再对上一步得到的特征做线性变换，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/38.png)

再WCT之后，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/39.png)

输入到decoder得到最终风格变换后的图像。




2018.07.16

Optical Flow

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




Face Alignment（一）

《Face Alignment by Explicit Shape Regression》

(1)借鉴了《Cascaded pose regression》的算法，给定初始Shape，使用shape-indexed特征，训练级联回归子，每个回归子的输出是Shape调整的偏移量；

(2)选择初始Shape，train的时候，从训练集的所有groundtruth中选择，test的时候，从训练集的groundtruth中挑选有代表性的作为标准Shape；

(3)two levels boosted regressiors，internal-level regression R和由所有R组成的external-level regression；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/45.png)

选择一个初始Shape，每一个Stage的R去拟合当前的Shape和groundtruth的偏差转换到Mean Shape空间的值(通过normalized shape操作)，把拟合的结果再转换到Shape空间；分成多个Stage比用一个Stage效果好，因为每个Stage使用的Shape Indexed Feature是在上一个Stage得到的Shape做相对的位置偏移得到的，随着Stages输出的Shape越来越准确，下一个Stage使用的特征也越来越准确；

(4)internal-level regression R，由K个fern组成，每个fern从400特征的差值(160000个特征)里选择5个作为特征(包括5个thresholds)，输出为32(2的5次方)个bin，每一个bin里的输出是整个Shape的编译；因为每一个fern的每一个bin都是训练集的groundtruth和初始Shape的加权之后的结果，所以每一个R的输出都是groundtruth和初始Shape的加权和，保证了最终拟合的Shape在训练集的Shapes组成的线性空间里(因为normalized shape操作只有scale和rotation)；

(5)Shape Indexed (Image) Features，两个像素点的插值作为特征，而且两个像素点的坐标使用相对于距离最近的关键点的相对坐标，选取特征时，根据regression target和特征之间的correlation来选取5个特征。

《Deep Convolutional Network Cascade for Facial Point Detection》

Cascaded Convolutional Network，不断精确关键点的位置，level1回归整张脸、眼睛+鼻子、鼻子+嘴的位置，level2-leveln对每个关键点在前一个level回归的位置上逐步选取更小的邻域作为下一个level的输入;

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/46.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/47.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/67.png)

《Facial Landmark Detection by Deep Multi-task Learning》

把关键点检测和pose, gender, wear glasses, smiling结合起来作多任务学习;

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/48.png)




2018.07.22

SeetaFaceEngine

SeetaFace Detection，《Funnel-Structured Cascade for Multi-View Face Detection with Alignment-Awareness》：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/49.png)

(1)先用LAB特征【1】+boosted cascade classifiers粗略检测人脸，把确定不是人脸的区域排除掉，其中，对于不同视角的人脸，分别训练一个分类器；

(2)SURF特征+粗略的MLP分类器，从多个LAB分类器输出的窗口输入到一个MLP cascade classifier，所以有若干个MLP cascade classifiers，其中一个MLP cascade classifier是多个MLP级联起来，每个MLP使用的特征的个数和网络的size逐渐增加，使用group sparse【2】方法挑选每个stage使用的SURF特征，只有通过上一个MLP才会输入到下一个MLP继续判断；

(3)shape-indexed特征+精细的MLP分类器，只有一个MLP cascade classifier，提取关键点位置的SIFT特征输入到MLP，输出人脸判断和关键点位置的调整，通过人脸判断的再传入到下一个MLP。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/50.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/51.png)

附：

【1】《Locally Assembled Binary (LAB) Feature with Feature-centric Cascade for Fast and Accurate Face Detection》

Locally Assembled Binary (LAB) Haar feature相当于二值化的HAAR特征,跟LBP的区别是:LBP是两个像素点的差值的二值化，LAB是两个区域的像素点的累加值的差的二值化,

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/52.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/53.png)

Feature-centric cascade: 对于一个窗口提取到的多维特征，不是一次全部输入到一个分类器里，而是分stage逐步增加特征的维度，直到最后一个stage才是整个维度的特征,以此来减少计算量。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/54.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/55.png)

【2】待补充



SeetaFace Alignment，《Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment》：

先用glabol SAN(Stacked Auto-encoder Networks)以原图片作为特征回归初始关键点位置，再用local SAN调整关键点位置，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/56.png)

(1)global SAN，用图片原像素值作为特征，训练自编码网络，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/57.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/58.png)

其中，每一层的权重的初始值通过预训练逐层得到，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/59.png)

(2)local SANs，使用Shape-indexed特征(例如SIFT)作为输入，得到关键点的调整值，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/60.png)

每个local SAN使用的特征的分辨率逐渐增加。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/61.png)

SeetaFace Identification，《VIPLFaceNet: An Open Source Deep Face Recognition SDK》：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/62.png)

使用FC2作为特征判断相似度。




SeetaFaceEngine2

Detection(无)


Alignment， 《A Fully End-to-End Cascaded CNN for Facial Landmark Detection》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/63.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/64.png)

H为整个network，S0是初始shape，可以使用mean shape，也可以通过cnn回归得到，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/65.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/66.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/68.png)


Identification(无)



2018.08.07

Face Alignment（二）

《Face Alignment at 3000 FPS via Regressing Local Binary Features》

(1)给定初始Shape，使用局部的二值特征生成特征向量，再用全局的线性回归得到Shape的偏移量，之所以使用局部特征而不是全局特征，是因为整张图片的候选特征太多，而且存在很多噪音，大多数有判决力的纹理信息都在人脸关键点的局部周围，而且关键点的位置和局部的纹理提供了充分的信息；

(2)对于每一个keypoint，在每个Stage里,先随机选取500个Shape-Indexed像素差值的二值化作为特征，训练random forest，把forest里所有的trees的leaves节点组成一个索引向量；

(3)把所有keypoints的索引向量链接起来作为特征，训练线性回归得到每个Stage的Shape的变化，因为线性回归的权重是所有训练集的Shapes的线性组合，所以最终拟合的Shape就是初始Shape和所有训练集的Shapes的线性组合，保证了Shape的约束。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/69.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/70.png)

《One Millisecond Face Alignment with an Ensemble of Regression Trees》

把ESR里的boost fern改成GBDT。


图片模糊模型：
 
《Digital Image Restoration》

Image degradation system

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/71.png)

(1)Motion Blur：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/72.png)

(2)Atmospheric Turbulence Blur：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/73.png)

(3)Uniform Out-of-Focus Blur：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/74.png)

(4)Uniform 2-D Blur：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/75.png)




2018.08.23

Face Alignment（三）

《Deep Recurrent Regression for Facial Landmark Detection》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/76.png)

(1)先使用conv-deconv神经网络输出特征点的heat map，每个特征点对应一个channel，从训练集合的样本中使用k-mean方法选出N个shapes作为候选，根据heat map输出的特征点位置找出最接近的候选shape作为初始shape；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/77.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/78.png)

(2)提取Deep Shape-indexed Features，在1中的网络的最后一个feature map上,以每个关键点为中心，在b*b窗口上上做max pooling，concentrate所有channels的结果作为特征；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/79.png)

(3)以Deep Shape-indexed Features为输入训练LSTM，回归shape的increment，在LSTM的每一个step,以调整后的shape重新提取Shape-indexed Feature作为输入。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/80.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/81.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/82.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/83.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/84.png)



《Approaching human level facial landmark localization by deep learning》

(1)First Level CNN：

用整张人脸回归出所有landmark的初始位置，根据初始位置对图片做相似变换,变换到标准shape；

(2)Second Level CNN：

每个landmark训练一个CNN，以landmark的局部区域作为输入，输出这个landmark临近的K个landmark的位置，最终每个lankmark的位置为所有CNN输出的平均值。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/85.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/86.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/87.png)


《Deep Alignment Network: A convolutional neural network for robust face alignment》

使用整张图片提取特征，给定一个初始Shape，级联式回归关键点位置，每一个Stage的回归子是一个CNN，回归子的输出为landmark调整的偏移量，第一个Stage以原图片作为输入，后面的Stage以变换过的图片、landmark heatmap、previous stage feature的concat作为输入；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/88.png)

(1)每个Stage的CNN：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/89.png)

(2)归一化到标准Shape：除了Stage1，每个Stage根据当前Shape和标准Shape之间的相似变换对输入图片做归一化；

(3)Landmark heatmap：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/90.png)

(4)Feature image layer：用前一个Stage的fc1层全连接到一个56*56的feature，再upscale到112*112；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/91.png)


《Mnemonic Descent Method: A recurrent process applied for》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/92.png)

使用整张图片作为输入，给定初始Shape，使用CNN提取特征,在CNN之后用RNN输出Shape的偏移量，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/93.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/94.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/95.png)

h为RNN的hidden state。


《Supervised Descent Method and its Applications to Face Alignment》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/96.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/97.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/98.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/99.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/100.png)

给定初始Shape，根据当前Shape的SIFT特征得到每一步的R、b，从而算出Shape的偏移量。




2018.09.09

Image Feature

《A COMBINED CORNER AND EDGE DETECTOR》（1988）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/101.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/102.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/103.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/104.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/105.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/106.png)

Treat gradient vectors as a set of (dx，dy) points with a center of mass defined as being at (0，0)，Fit an ellipse to that set of points via scatter matrix，Analyze ellipse parameters for varying cases，M矩阵类似于对(dx，dy)做PCA；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/107.png)

Harris Detector对旋转变换不变，但对scale变换改变。



《Local Grayvalue Invariants for Image Retrieval》（1997）

(1)使用Harris Detector检测关键点；

(2)特征描述：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/108.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/109.png)

i、j、k、l遍历x1、x2两个维度，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/110.png)

描述子对scale变换保持不变。


《Matching Images with Different Resolutions》（2002）

(1)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/111.png)

m为低分辨率图片，n为高分辨率图片，假设没有旋转，在两个分辨率的图片上分别计算Harris Detector，根据

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/112.png)

可得

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/113.png)

对高分辨率图片使用scale space，令s = 1 / h，可得

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/114.png)

(2)对于高分辨率图片，取不同的scale值，计算Local Jet描述子，最高到3阶，采用局部仿射不变的形式(见《Detection of local features invariant to affines transformations》)，共7维，和低分辨率图片的描述子比较，使用Mahalanobis距离，找到匹配度最高的。


《Detection of local features invariant to affines transformations》(2002)

(1)特征点检测

Harris-Laplace detector；

仿射不变特征：TODO

(2)描述子

Local Jet：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/115.png)

可以使用Local Jet的方向导数作为描述子，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/116.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/117.png)

应该为v[0，……，11]，仿射不变，或者使用Local Jet的组合作为描述子，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/118.png)

这个描述子对旋转不变，如果去掉前两个分支，并且剩余的分支除以第二个分支的合适的幂，则描述子对仿射不变。



《Indexing based on scale invariant interest points》（《Detection of local features invariant to affines transformations》的部分内容，2001）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/119.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/120.png)

(1)Harris detector在一个scale内表现较好，在不同的scales之间不鲁邦，所以在2D空间上，使用Harris detector找到极大值点，且大于门限值，作为候选点，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/121.png)

(2)在不同的scales之间，Laplacian detector的极大值点具有鲁棒性，所以在3D空间上，使用Laplacian detector从候选点中找到极大值点，且大于门限值，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/122.png)

(3)使用Local Jet的方向导数作为描述子，在关键点的邻域求导数，取直方图的峰值，最高为4阶导数，且除以一阶导数，共12维；

(4)使用Mahalanobis distance比较描述子。


《Scale & Affine Invariant Interest Point Detectors》（2004）

The DoG detector detects mainly blobs, whereas the Harris detector responds to corners and highly textured points, hence these detectors extract complementary features in images.

TODO


《Distinctive Image Features from Scale-Invariant Keypoints》（SIFT， 2004）

(1)Point Detector

difference-of-Gaussian function(approximation to the scale-normalized Laplacian of Gaussian)，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/123.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/124.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/125.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/126.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/127.png)

Laplacian Image为L(x,y,σ)，L(x,y, kσ)，L(x,y,k^2 * sigma), …… ,L(x,y,k^s * sigma)，L(x,y,k^(s+1) * sigma)，L(x,y,k^(s+2) * sigma)，相邻的两个L相减得到一个octave里的D，下一个octave里第一个L是对L(x,y,k^s * sigma)做下采样，在得到的difference-of-Gaussian scale space（即所有D）上的3*3*3邻域中寻找极值点；

(2)准确的keypoint定位

以采样点为原点做Taylor展开，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/128.png)

要求出D(x)的极值点，上式对x求导，令结果=0，求出x的置，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/129.png)

极值为

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/130.png)

如果此值的绝对值小于0.03，则丢弃这个关键点；

(3)消除边的响应

Hessian matrix的特征值和主曲率成比例，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/131.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/132.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/133.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/134.png)

α、β为特征值，且α>=β，α=rβ，因为边的主曲率的ratio值比较大，所以利用Tr(H)平方和Det(H)的比值，消除ratio大于r的keypoints（r=10）；

(3)方向判定

为了保持特征旋转不变，所以要给关键点确定一个主要的方向。选择距离keypoint被检测到的scale最近的Gaussian光滑图像，计算keypoint邻域里的采样点的梯度大小和方向，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/135.png)

计算36个bins的梯度方向直方图，每一个采样点对直方图的分配为梯度大小乘以标准差为keypoint的scale的1.5倍的Gaussian权重，选择直方图的峰值作为方向，在最高峰值的80%以内的方向也选为方向，所以同一个keypoint会生成多个同样位置和scale、但不同方向的keypoint，用最接近峰值的3个直方图值来插值，得到峰值最终的准确位置；

(4)局部图像描述子

类似于(3)，选择keypoint最近的scale space Gaussian模糊图像，采样梯度的大小和方向。描述子的坐标和梯度方向要相对(3)中计算出的keypoint方向旋转，使用描述子窗口宽度1.5倍标准差的Gaussian权重函数作为采样点梯度大小的权重。在4x4区域里计算梯度直方图，直方图的方向分成8个bin，共计算4x4个区域，所以描述子的维度为4x4x8=128。在计算每个采样点的梯度方向对bin的分配时，要使用trilinear，除了空间上的bilinear，还有bin上的linear：每一个梯度会分配到梯度方向和bin中心方向值最近的两个bins，分配值大小为梯度大小乘以(1-d)，d为梯度方向和bin中心方向在bin空间单位化后的距离。得到的描述子向量先归一化为单位向量，对于大于0.2的分量限制为0.2，再归一化为单位向量；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/136.png)

(5)关键点匹配
使用Best-Bin-First（BBF）【1】算法查找特征的近邻。

注【1】

《Shape indexing using approximate nearest-neighbour search in highdimensional spaces》

TODO

《SURF: Speeded Up Robust Features》（2006）

思路：(1)基于Hessian的检测子要比相匹配的基于Harris的检测子更稳定和具有可重复性，使用Hessian矩阵的行列式比使用Hessian矩阵的迹（Laplacian）更有优势；(2)使用DoG类似的近似可以提高速度，但准确度损失很少；

(1)Fast-Hessian Detector

Hessian矩阵的定义为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/137.png)

使用长方形区域为常值的box filter近似Gaussian导数，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/138.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/139.png)

在图片空间和Scale空间的3*3*3邻域里寻找Hessian矩阵行列式的最大值，使用SIFT中类似的方法插值出sub-pixel、sub-scale的Hessian矩阵行列式的最大值；

(2)描述子

（2.1）方向判决

在关键点半径为6s的圆形邻域里，计算Haar-wavelet响应，s为keypoint被检测到的scale，采样的步骤和Haar-wavelet响应的计算也在当前scale进行，使用积分图代替wavelet， wavelet的边长为4s.得到wavelet响应后，用以关键点为中心，标准差为2.5s的Gaussian核加权.以覆盖π/3的角度滑动窗口，计算窗口内的x、y方向wavelet响应的和为新的向量，最长向量的方向为关键点的方向；

(2.2)描述子组成

选取兴趣点20sx20s的正方形邻域，方向为(2.1)确定的方向，分成4x4的子区域，对于每个子区域，在5x5的均匀的空间采样点上计算水平方向和垂直方向的Haar wavelet，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/163.png)

filter size是2s，the responses乘以以兴趣点为中心的、标准差为3.3s的Gaussian权重。在每个子区域里计算wavelet response的和，每个子区域的描述子为

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/164.png)

归一化为单位向量，最终的描述向量为64维。

Upright版本的SURF(U-SURF)不需要方向判定，在计算描述子时，选取垂直方向的正方形邻域。

在对兴趣点做匹配时，Laplacian的符号(即Hessian矩阵的迹)也要包含进去，只匹配符号相同的兴趣点。



《Machine learning for high-speed corner detection》（2006）

(1)FAST: Features from Accelerated Segment Test

点p的周围一圈如果有连续n个点的像素值比p点的像素值大于t，或者小于t，则p是一个corner，n的一个常用值为12，这样可以先只检查1、5、9、13四个点：如果p是一个corner，则四个点里至少有三个满足上述条件；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/140.png)

(2)Machine learning a corner detector

根据p点和近邻x点的像素值，以及门限t，可以把p分为三组：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/141.png)

以熵作为判断规则：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/142.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/143.png)

选择合适的x和t，把p分成三个子集后，在每个子集中继续重复上述步骤，直到一个子集的熵为0，最终形成一棵decision tree；

(3)Non-maximal suppression

对每一个选出的corner赋予一个得分V，相邻的corners，丢弃V的值低的，V的几种选择：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/144.png)

为了计算速度，最终使用下式计算V：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/145.png)

其中，V的值使用p的邻域里所有的x计算。


《CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching》（2008）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/146.png)

使用bi-level center-surround filters近似Laplacian，bi-level意思是filter的每个权重值为1或者-1，不同size的filter对应不同的scale，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/147.png)

(1)CenSurE Using Difference of Boxes

I为内部box的权重，O为外部box的权重，I和O满足：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/148.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/149.png)

block size n取值为[1,2,3,4,5,6,7]，1和7为边界，所以最低的scale对应block size 2，对应的LoG的sigma近似为1.885，5个中间的scales覆盖了2.5个octaves；

(2)CenSurE Using Octagons

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/150.png)

与box类似，要找到DC response是0的权重I和O，(DC response:对于常数区域的响应？)；

(3)非极大抑制

得到filters的响应后，在3x3x3的邻域内找到极值点，过滤掉那些响应小于门限值的点，因为所有的filters都是在原图片上计算，所以不需要对图片进行subsample；

(4)直线抑制

计算Harris measure，在block size 2上使用9x9的window和门限值10，window的大小和scale线性相关，scale越高,window越大，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/151.png)

(5)filter计算

使用integral image计算filter response：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/152.png)

α为倾斜角度，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/153.png)

(6)Modified Upright SURF (MU-SURF) Descriptor

As pointed out by David Lowe（《Distinctive image features from scale-invariant keypoints》），“it is important to avoid all boundary effects in which the descriptor abruptly changes as a sample shifts smoothly from being within one histogram to another or from one orientation to another.”

兴趣点的邻域size从20s增加为24s，增加2s的pad，构造summed image，summed image每个像素点的值是sxs区域的像素和，在区域中的24x24个点上计算filter size为2s的水平方向和垂直方向的Haar wavelet，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/163.png)

因为Haar wavelet的filter size是2s，所以可以分成4个sxs区域，垂直方向和水平方向的Haar wavelet就是4个区域的像素和加或减。

把区域分成4x4个正方形子区域，每个区域size为9x9，区域之间有2个像素重合（Why?），在每个子区域中，response的值乘以以子区域中心点为中心、标准差为2.5的Gaussian权重，求和得到每个子区域的描述向量，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/164.png)

每个子区域的描述向量再乘以以特征点为中心、标准差为1.5、size为4x4的Gaussian权重mask，得到的向量归一化为单位向量。





《BRIEF: Binary Robust Independent Elementary Features》（2010）

(1)关键点检测

使用FAST（《Machine learning for high-speed corner detection》）或者CenSurE（《CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching》）方法检测；

(2)描述子

在关键点9x9的邻域里，先使用方差为2的高斯核平滑，再选取n对点，把每对点的差值二值化，二值化得到的0或1连接成特征向量，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/154.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/155.png)

每对点选取的方法为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/156.png)

使用Hamming distances对关键点特征进行匹配；

(3)BRIEF特征不是旋转不变的。




《ORB: an efficient alternative to SIFT or SURF》（2011）

(1)关键点检测及方向判断

使用半径为9的FAST Detector检测关键点，再使用Harris corner measure过滤掉不是corner的关键点。在关键点的patch区域里使用intensity centroid方法判断每个关键点的方向，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/157.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/158.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/159.png)

atan2为四个象限的arctan；

(2)关键点描述子

使用BRIEF做为特征，为了满足旋转不变性，使用关键点的方向计算Steered BRIEF，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/160.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/161.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/162.png)

(x,y)为提取二值test点的位置，R为关键点方向的旋转矩阵，其中关键点的方向被离散化为12°的间隔，这样可以预计算旋转后的位置。

但是Steered BRIEF的variance更低，所以使用以下方法选取test点的位置：选取关键点的31x31邻域作为patch，每一个test是5x5的子窗口，所以共有205590中可能的test，在训练集上计算所有的test，根据test的平均值和0。5的距离从小到大排序， 构成向量T，把T中第一个test放入结果向量R中，并从T中移除，继续从T中取出test，计算该test和R中的test的相关性，如果大于门限值，则丢弃，否则放入R，重复这个过程最终得到256维的R，如果R中的test个数小于256，则提高相关性门限值重新尝试；

(3)二值特征匹配

使用multi-probe Locality Sensitive Hashing（【2】）进行最近邻搜索。


注【2】

《Multi-probe LSH: efficient indexing for high-dimensional similarity search》

TODO



《Adaptive and Generic Corner Detection Based on the Accelerated Segment Test》（AGAST, 2010）

TODO


《DAISY: An Efficient Dense Descriptor Applied to Wide-Baseline Stereo》（2010）

TODO


《Brisk: Binary robust invariant scalable keypoints》（2011）

(1)	ScaleSpace Keypoint Detection

n个octaves c和n个intra-octaves d，通常n=4，原始图片对应c0，其余的c逐步地half-sampling，每个d(i)在c(i)和c(i+1)之间，d0是c0的1.5倍下采样。

使用FAST 9-16检测子，在每一个octave和intra-octave上使用同样的门限T找到潜在的兴趣点，在scale-space上使用NMS：首先，点要满足相比于同一layer上的8个邻域点的FAST score是极大值，score定义为判定一个点是corner的最大门限，其次，上一层和下一层的layer的scores也要更小。在每一层上检查size相等的正方形patch：在极大值被检测到的layer上边长为2个像素(即3x3个点)。相邻的layers使用插值得到FAST scores。

在3个scores-patches上的3x3区域里使用2D二次函数差值，得到三个layers的子像素的score极大值，沿着scale坐标轴拟合1D抛物线得到最终的score和scale。最后，根据得到的scale在相邻的两个layers的patches上重新插值得到极大值点的坐标。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/165.png)

(2)	Keypoint Description

(2.1) Sampling Pattern and Rotation Estimation

关键点邻域的采样模式为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/166.png)

对于每个采样点p，使用Gaussian光滑，标准差正比于点p与关键点的距离。对于N个采样点，共有N*(N-1)/2个采样点对，局部梯度定义为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/167.png)

I为光滑后的像素值。

考虑所有采样点对集合

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/168.png)

根据距离定义两个子集short-distance pairings S和long-distance pairings L：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/169.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/170.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/171.png)

t为关键点的scale，估计关键点的方向为

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/172.png)

(2.2) Building the Descriptor

使用(2.1)中估计的方向，在关键点邻域采样，描述子为所有short-distance pairings S的像素值的比较：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/173.png)

BRISK与BRIEF的不同：BRISK使用确定的采样模式导致一致的采样点；使用裁剪过的Gaussian光滑，不会破坏接近的两个比较点的像素信息；BRISK使用更少的采样点，即一个采样点参与多次比较；在空间上限制像素点的比较。

根据采样模式和距离的门限，最终得到的bit-string长度为512。

(3)	Descriptor Matching

使用Hamming distance比较描述子，即两个描述子中不同的bit的数量。














