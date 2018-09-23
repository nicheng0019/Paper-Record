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


《Face Alignment at 3000 FPS via Regressing Local Binary Features》

(1)给定初始Shape，使用局部的二值特征生成特征向量，再用全局的线性回归得到Shape的偏移量，之所以使用局部特征而不是全局特征，是因为整张图片的候选特征太多，而且存在很多噪音，大多数有判决力的纹理信息都在人脸关键点的局部周围，而且关键点的位置和局部的纹理提供了充分的信息；

(2)对于每一个keypoint，在每个Stage里,先随机选取500个Shape-Indexed像素差值的二值化作为特征，训练random forest，把forest里所有的trees的leaves节点组成一个索引向量；

(3)把所有keypoints的索引向量链接起来作为特征，训练线性回归得到每个Stage的Shape的变化，因为线性回归的权重是所有训练集的Shapes的线性组合，所以最终拟合的Shape就是初始Shape和所有训练集的Shapes的线性组合，保证了Shape的约束。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/69.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/70.png)


《One Millisecond Face Alignment with an Ensemble of Regression Trees》

把ESR里的boost fern改成GBDT。


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
