《A COMBINED CORNER AND EDGE DETECTOR》（1988）

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/101.png)

MORAVEC方法：寻找偏移(x，y)，使得在一个长方形窗口内偏移后的图像像素和原图片像素的差的平方和最小，(x，y)只考虑固定几个方向。

Harris方法的改进：

(1)MORAVEC方法只考虑几个间隔45度的固定方向，Harris方法通过Taylor展开可以覆盖所有方向的小的偏移：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/102.png)

(2)MORAVEC方法使用长方形窗口且所有像素的权重一样，所以对噪音敏感，Harris方法使用圆形光滑的窗口，例如Gaussian权重：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/180.png)

(3)MORAVEC方法容易对边产生响应，Harris方法使用E在偏移方向上的变化来测量corner：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/103.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/104.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/105.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/106.png)

Treat gradient vectors as a set of (dx，dy) points with a center of mass defined as being at (0，0)，Fit an ellipse to that set of points via scatter matrix，Analyze ellipse parameters for varying cases，M矩阵类似于对(dx，dy)做PCA；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/107.png)

E是点的邻域的梯度的加权和矩阵。

当前点的方向导数是梯度乘以方向角度的cos、sin值。

判断flat、edge、corner，即判断所有方向导数中，幅度的最大值和最小值，而E的特征值恰好对应这个最大值和最小值。

PCA即近似变化率最大的方向，在这里变化最大即导数幅度最大。

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

在两张图片的scale变换已知的情况下,使用调整的Harris检测子得到两张图片对应的关键点。

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

仿射不变特征：见《An affine invariant interest point detector》；

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

(1)Harris detector在一个scale内表现较好，在不同的scales之间不鲁棒，所以在2D空间上，使用Harris detector找到极大值点，且大于门限值，作为候选点，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/121.png)

(2)在不同的scales之间，Laplacian detector的极大值点具有鲁棒性，所以在3D空间上，使用Laplacian detector从候选点中找到极大值点，且大于门限值，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/122.png)

(3)使用Local Jet的方向导数作为描述子，在关键点的邻域求导数，取直方图的峰值，最高为4阶导数，且除以一阶导数，共12维；

(4)使用Mahalanobis distance比较描述子。


《An affine invariant interest point detector》 (2002)

It is based on three key ideas: 1) The second moment matrix computed in a point can be used to normalize a region in an affine invariant way (skew and stretch). 2) The scale of the local structure is indicated by local extrema of normalized derivatives over scale. 3) An affine-adapted Harris detector determines the location of interest points.

(1) Affine Gaussian scale-space

在几个scale上使用Harris measure：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/194.png)
 
在uniform Gaussian上，自动scale选择基于正则化的Laplacian的最大值：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/195.png)
 
在仿射scale空间上，non-uniform Gaussian核定义为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/196.png)
 
在non-uniform scale空间上，二阶moment矩阵定义为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/197.png)


《Robust Wide Baseline Stereo from Maximally Stable Extremal Regions》 (2002)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/198.png)

让某个极值区域的面积在序列中变化率最小的像素level作为该区域的threshold，位置和threshold同时作为一个MSER的输出；

TODO


《Scale & Affine Invariant Interest Point Detectors》（2004）

The DoG detector detects mainly blobs, whereas the Harris detector responds to corners and highly textured points, hence these detectors extract complementary features in images.

不同的detectors的比较：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/222.png)

Harris-Laplace：同一个scale上使用Harris，不同的scale-space之间使用Laplace（《Indexing based on scale invariant interest points》）；

TODO

《Object Recognition from Local Scale-Invariant Features》 (1999), 《Distinctive Image Features from Scale-Invariant Keypoints》（2004），（SIFT） 

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

类似于(3)，选择keypoint最近的scale space Gaussian模糊图像，采样梯度的大小和方向。描述子的坐标和梯度方向要相对(3)中计算出的keypoint方向旋转，使用描述子窗口宽度1.5倍标准差的Gaussian权重函数作为采样点梯度大小的权重。在4x4区域里计算梯度直方图，直方图的方向分成8个bin，共计算4x4个区域，所以描述子的维度为4x4x8=128，总的邻域大小为16x16，因为是用相邻两个像素算的是x、y方向的梯度，所以相当于是在关键点的17x17的邻域。在计算每个采样点的梯度方向对bin的分配时，要使用trilinear，除了空间上的bilinear，还有bin上的linear：每一个梯度会分配到梯度方向和bin中心方向值最近的两个bins，分配值大小为梯度大小乘以(1-d)，d为梯度方向和bin中心方向在bin空间单位化后的距离。得到的描述子向量先归一化为单位向量，对于大于0.2的分量限制为0.2，再归一化为单位向量；

（1999)中除了在当前的scale里计算4x4x8维特征向量之外，还在更高一个octave的scale里的2x2邻域里按同样方式计算8个bins，所以总的特征向量为160维。

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/136.png)

(5)关键点匹配
使用Best-Bin-First（BBF）【1】算法查找特征的近邻。

注【1】

《Shape indexing using approximate nearest-neighbour search in highdimensional spaces》

TODO

《A performance evaluation of local descriptors》（2005）

A. Support regions

1）Region detectors

Harris points：旋转不变；

Harris-Laplace regions：旋转和scale不变，检测corner-like结构；

Hessian-Laplace regions：旋转和scale不变，检测blob-like结构，DoG对edge也会有相应；

Harris-Affine regions：仿射不变；

Hessian-Affine regions：仿射不变。

B. Descriptors

SIFT descriptors：见《Object Recognition from Local Scale-Invariant Features》；

Gradient location-orientation histogram (GLOH)：使用极坐标，在半径方向上分为3个bins，在角度方向上分为8个bins，共17个bins。梯度方向分为16个bins，共272个bins，使用PCA得到128维的特征；

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/223.png)

Shape context：

TODO

《Multi-Image Matching using Multi-Scale Oriented Patches》




《SURF: Speeded Up Robust Features》（2006）

思路：(1)基于Hessian的检测子要比相匹配的基于Harris的检测子更稳定和具有可重复性，使用Hessian矩阵的行列式比使用Hessian矩阵的迹（Laplacian）更有优势；(2)使用DoG类似的近似可以提高速度，但准确度损失很少；

(1)Fast-Hessian Detector

Hessian矩阵的定义为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/137.png)

使用长方形区域为常值的box filter近似Gaussian导数，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/138.png)

计算区域的Gaussian导数矩阵和box filter的Frobenius范数，可得：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/139.png)

第一个octave：

第一层：9x9 box filter，对应σ=1.2，
 
第二层：15x15 box filter，对应σ=1.2x15/9，
 
第三层：21x21 box filter，……
 
第四层：27x27 box filter，……
 
……

第二个octave：

第一层：18x18 box filter，对应σ=1.2x18/9，

第二层：30x30 box filter，对应σ=1.2x30/9，
	
第三层：42x42 box filter，……
	
第四层：54x54 box filter，……
	
……

第三个octave：
	
第一层：36x36 box filter，对应σ=1.2x36/9，
	
……


在图片空间和Scale空间的3*3*3邻域里寻找Hessian矩阵行列式的最大值，使用SIFT中类似的方法插值出sub-pixel、sub-scale的Hessian矩阵行列式的最大值；

(2)描述子

（2.1）方向判决

在关键点半径为6s的圆形邻域里，计算Haar-wavelet响应，s为keypoint被检测到的scale，采样的步长为s，使用积分图代替wavelet， wavelet的边长为4s。得到wavelet响应后，用以关键点为中心，标准差为2.5s（2.5s的意思是标准差为2.5的Gaussian权重，以s为间隔采样的像素值与权重相乘）的Gaussian核加权。以覆盖π/3的角度滑动窗口，计算窗口内的x、y方向wavelet响应的和为新的向量，最长向量的方向为关键点的方向；

(2.2)描述子组成

选取兴趣点20sx20s的正方形邻域，方向为(2.1)确定的方向，分成4x4的子区域，对于每个子区域，以s为步长采样，得到在5x5的均匀的采样点，计算相对于主方向的水平方向和垂直方向的Haar wavelet，

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/163.png)

Haar wavelet区域边长为2s，the responses乘以以兴趣点为中心的、标准差为3.3s（与2.5s的意思相同）的Gaussian权重。在每个子区域里计算wavelet response的和，每个子区域的描述子为

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

在关键点9x9的邻域里，先使用方差为2的高斯核平滑，再在邻域里选取n对点（n=128，256，512），把每对点的差值二值化，二值化得到的0或1连接成特征向量，

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

但是Steered BRIEF的variance更低，所以使用以下方法选取test点的位置：选取关键点的31x31邻域作为patch，每一个test是5x5的子窗口，所以共有205590种可能的test，在训练集上计算所有的test，根据test的平均值和0.5的距离从小到大排序（均值越接近0.5，说明变化越大，特征越具有判决力）， 构成向量T，把T中第一个test放入结果向量R中，并从T中移除，继续从T中取出test，计算该test和R中的test的相关性，如果大于门限值，则丢弃，否则放入R，重复这个过程最终得到256维的R，如果R中的test个数小于256，则提高相关性门限值重新尝试；

(3)二值特征匹配

使用multi-probe Locality Sensitive Hashing（【2】）进行最近邻搜索。


注【2】

《Multi-probe LSH: efficient indexing for high-dimensional similarity search》

TODO



《Adaptive and Generic Corner Detection Based on the Accelerated Segment Test》（AGAST, 2010）

在FAST算法中，构造一棵三叉树，原点p的每个相邻点x有四种状态：unknown(u)， darker(d)， brighter(b) or similar(s)，每个学习的步骤使用两个问题：“is brighter” and “is darker”。

新加两种状态：“not brighter” and “not darker”。训练一棵二叉决策树，树的每个节点选择一个邻域点x和一个问题：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/174.png)




《DAISY: An Efficient Dense Descriptor Applied to Wide-Baseline Stereo》（《A Fast Local Descriptor for Dense Matching》）（2010）

TODO



《Brisk: Binary robust invariant scalable keypoints》（2011）

(1)	Scale Space Keypoint Detection

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

所以BRISK是旋转不变的。

(2.2) Building the Descriptor

使用(2.1)中估计的方向，在关键点邻域采样，描述子为所有short-distance pairings S的像素值的比较：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/173.png)

BRISK与BRIEF的不同：BRISK使用确定的采样模式导致一致的采样点；使用裁剪过的Gaussian光滑，不会破坏接近的两个比较点的像素信息；BRISK使用更少的采样点，即一个采样点参与多次比较；在空间上限制像素点的比较。

根据采样模式和距离的门限，最终得到的bit-string长度为512。

(3)	Descriptor Matching

使用Hamming distance比较描述子，即两个描述子中不同的bit的数量。


《FREAK: Fast Retina Keypoint》（2012）

(1)Retinal sampling pattern

感知域（receptive fields）的拓扑结构为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/175.png)

每个圆表示对应采样点的Gaussian核的标准差；

(2)Coarse-to-fine descriptor

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/176.png)

P是一对感知域，N是描述子的长度， 

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/177.png)

I是感知域对P光滑后的像素值。使用类似ORB中的方法找到描述子的感知域对:求出所有关键点的所有感知域对的二值特征；求出每一个感知域对的均值，选出变化最大（即均值最接近0.5）的感知域对；依次从剩下的二值特征里选择和已选的特征相关性最小且均值最接近0.5的；

(3)Saccadic search

搜索的时候，先比较描述子的钱16bytes，如果距离小于门限值，再继续比较下一个byte；

(4)Orientation

用来计算局部梯度的所有对为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/178.png)

设G为所有用来计算梯度的感知域对，则方向为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/179.png)

M为G中对的个数。


《LATCH: Learned Arrangements of Three Patch Codes》 (2015)

普通的二值特征：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/199.png)
 
W(p，σ)为窗口W中的点p经过高斯滤波平滑后的像素值，最终得到的特征向量为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/200.png)
 
patch triplets的二值特征为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/201.png)
 
P为kxk的patch。

学习patch triplet arrangement：

在训练数据中选出500000对image patches，一半是从不同角度观察到的同一物理场景的点，标记为“same”，另一半为不同场景的点，记为“not-same”。随机选取p(t)，p1，p2的坐标，生成56000个triplet arrangement，在所有patch pairs上算出每一个arrangement的二值特征，定义一个arrangement的quality为在“same”的pairs上得到同样二值的个数加上在“not-same”的pairs上得到不同二值的个数。按照quality从大到小排列，选取quality值最大、且和已选择的arrangements的correlation小于门限值(0.2)的arrangements。实际使用中，最终选取的arrangements不超过256个。

