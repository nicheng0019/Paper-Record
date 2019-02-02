《Categorizing Nine Visual Classes using Local Appearance Descriptors》 （2004）（《Visual Categorization with Bags of Keypoints》）

(1)Detection and description of image patches

使用Harris affine detector检测关键点，确定关键点的椭圆（仿射）邻域，把仿射邻域映射到圆形区域，在区域上提取SIFT特征作为描述子；

(2)Assigning patch descriptors to a set of predetermined clusters (a vocabulary) with a vector quantization algorithm

使用k-means方法得到vocabulary， k的值和聚类中心的初始值经过多次尝试，选择经验风险（empirical risk）最小的值；

(3)Constructing a bag of keypoints, which counts the number of patches assigned to each cluster

(4)Applying a multi-class classifier, treating the bag of keypoints as the feature vector, and thus determine which category or categories to assign to the image

使用朴素Bayes

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/226.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/227.png)

或者SVM

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/228.png)

做分类，SVM使用one-against-all的方法训练多类分类器，选择输出最大的作为分类结果。


《Video Google: A Text Retrieval Approach to Object Matching in Videos》 （2003）

(1)Viewpoint invariant description

使用《An affine invariant interest point detector》中的方法检测兴趣点，记为SA，使用像素值水浸图像分割方法检测近似稳定的区域，记为MS。SA检测的是角点，MS检测的是和周围形成高对比度的块状（例如灰色墙上的深色窗户）。两种类型的区域都用椭圆表示。在椭圆区域上提取128维的SIFT特征。每一帧中检测到的区域使用simple constant velocity dynamical model and correlation跟踪。存在不超过3帧的被抛弃掉。区域最终的描述使用跟踪过程中所有描述的均值。

(2)Building a visual vocabulary

diagonal covariance matrix最大的10%的区域被拒绝掉。使用K-means聚类的方法构造vocabulary。使用Mahalanobis距离，在所有数据上计算covariance矩阵。SA和MS区域分别聚类构造vocabulary，类的个数和SA\MS描述子的个数成正比。

(3) Visual indexing using text retrieval methods

分别使用binary、tf、tf-idf的方法作为词频率向量的权重，tf-idf权重的效果最好。


《Constructing visual models with a latent space approach》

使用Probablistic Latent Semantic Analysis方法构造模型，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/229.png)

使用Difference of Gaussians（DOG）作为关键点检测子，提取 SIFT特征，提取的关键点及其特征相当于word，aspect相当于hidden topic，image相当于document。使用EM拟合方法得到P(z|d)和P(x|z)。





