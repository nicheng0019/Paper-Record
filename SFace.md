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
