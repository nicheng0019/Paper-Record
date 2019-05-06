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

一个LAB特征可以用一个4元组l(x,y,w,h)标记，当前区域和相邻的8个区域比较大小，如Figure5，取值范围为[0,255]，论文里使用3x3的区域。

检测的时候，先计算整个图片的LAB特征，再用每个窗口的LAB特征做分类。分类器使用RealBoost，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/239.png)

T是使用的LAB特征总数，l(i)是某一个LAB特征，h对每一个LAB特征值返回一个置信度，并包含一个门限值。为了提高检测效率，把T个l分成若干stages，每个stage结束使用门限值判断是否为人脸，如果不是，则不需要之后的判断。

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
