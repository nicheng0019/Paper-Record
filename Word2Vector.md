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

