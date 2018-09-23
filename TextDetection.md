《Multi-Oriented Text Detection with Fully Convolutional Networks》

先用FCN得到候选的文字区域，再用传统方法分割出一行行文字和文字方向，再用FCN得到每个字符的中心，再进一步分类文字和非文字；

《Scene Text Detection via Holistic, Multi-Channel Prediction》

使用FCN生成三张map，一张分割行文字，一张分割字符，一张回归每个像素点对应的文字方向；

《Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection》

使用CNN+带方向的滑动窗口来回归文字区域，使用Monte-Carlo方法来计算不规则四边形的重合面积；

《EAST: An Efficient and Accurate Scene Text Detector》

使用FCN回归每个像素点是文字的score，以及对应的框的位置，再把框合并，对较长文字的检测效果不太好，两端会有漏掉的部分，可能时因为网络的Receptive Field太小。



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
