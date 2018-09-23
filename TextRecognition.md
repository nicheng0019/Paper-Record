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
