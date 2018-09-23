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
