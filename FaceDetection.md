《Multi-view Face Detection Using Deep Convolutional Neural Networks》

使用修改过的AlexNet，把FC层改为卷积层，图片大小不限制，图片最小尺寸大于227，最大放大5倍，每个octave缩放3次（octave：八度音，音乐上相邻的八度音的频率比为1：2，所以图像上相邻的octave的尺寸比为1：2），最后一层生成1张heatmap，AlexNet的输入图片的尺寸为227x227，所以heatmap上一个点对应原图像上一个227x227的区域，stride为32，根据heatmap上大于门限值的点得到对应的原图片上区域为人脸，再使用NMS方法过滤得到的区域，可以检测不同角度的人脸，但无法给出具体角度,

两种NMS方法：(1)NMS-max，即传统的NMS，两个boxes的IOU大于门限值，则删掉score小的box，(2)NMS-avg，根据IOU把boxes聚类，每个cluster里把score低的boxes删掉，取剩余的boxes的均值作为最终box的值，cluster里最大的score作为最终的score，EAST里使用的合并文字检测框的方法和这个比较像。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/18.png)


《DenseBox: Unifying Landmark Localization with End to End Object Detection》

全卷积网络，输出heatmap和左上\右下两个点的相对位置，类似于EAST算法，使用多个尺寸测试，输出的box位置回归为上下左右相对于当前点位置的偏移，且相对于图像尺寸归一化为[0, 1]。可以加入关键点位置训练来refine网络。


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


《CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/202.png)

(1) Identifying Tiny Faces

为了检测到tiny faces，使用multiple scales Faster-RCNN，把conv3、conv4、conv5的特征融合到一起，通过对conv3和conv4使用pooling来达到尺寸一致。

因为浅层的特征值通常偏大，深层的特征值偏小，为了在融合浅层和深层特征之后，避免浅层特征占主导，对每一层的特征分别做L2正则化：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/203.png)
 
D为channel的个数，即对同一层feature map的所有像素同时正则化。正则化会导致特征值取值变小，降低特征的判决力，所以对每一个channel再乘以一个scaling factor：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/204.png)
 
back-propagation时参数的更新：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/205.png)
 
(2) Integrating Body Context

在Multi-Scale Region Proposal Network proposal人脸区域后，根据区域算出人body的区域，在feature map上得到body的区域的特征，对body的特征做ROI pooling，变为和face特征同样的尺寸。分别对body和face在不同layers的特征做L2正则化，再分别融合进行特征提取，得到最后的fully connect特征，再把两个特征连接起来，得到最后的分类和位置回归。

最后融合body和face的特征比在ROI pooling得到body特征之后马上融合的效果要好。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/206.png)


《Faceness-Net: Face Detection through Deep Facial Part Responses》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/207.png)

(1)	Attribute-Aware Networks

训练5个神经网络，分别生成hair、eyes、nose、mouth and beard 5个区域的partness map。

Network structure：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/208.png)
 
Shared representation：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/209.png)
 
(2)	Generating Candidate Windows

Generic object proposal：

使用一般的目标区域生成方法，使用faceness score来对区域排序；

Template proposal：

对于每一个partness map，找出大于某个门限值的所有位置，在每个位置使用模板生成多个指定scale和aspect ratio的proposals。在每一个proposal上，计算Faceness Score。 Faceness Score由人脸不同parts的主区域的map值的和除以剩余区域的map值的和。每一个part的主区域跟part相关，通过训练得到。使用NMS合并proposals，选出每个partness中Score最高的proposal。再合并所有parts的proposals，再使用NMS合并proposals，选出score最高的windows。

(3)	Face Detection

使用Multi-task的refine神经网络对区域做分类和box回归。


《Scale-Aware Face Detection》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/210.png)

(1) Scale Proposal Network (SPN)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/211.png)

使用fully convolutional网络，输出n个bins的直方图，直方图的每个bin表示face的size在某个范围之内的概率：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/212.png)
 
s0直方图最左边的值，d为直方图的。

对于size为s的face，定义Gaussian function：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/213.png)
 
则第i个bin从f(x)的采样值为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/214.png)
 
(2) Scaling strategy generation

先使用移动平均平滑直方图，再使用一维的NMS。选择置信度大于某个门限值的proposals。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/215.png)
 
(3) Single-scale RPN

使用一个anchor的RPN来检测人脸。


《FaceBoxes: A CPU Real-time Face Detector with High Accuracy》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/216.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/217.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/218.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/219.png)


《Face Attention Network: An Effective Face Detector for the Occluded Faces》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/220.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/221.png)


