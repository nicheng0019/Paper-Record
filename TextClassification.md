1、最简单的方法：对文本中的每个word的representation vector做加权平均，得到的vector作为文本的vector；

2、《Distributed Representations of Sentences and Documents》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/15.png)
![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/16.png)

监督学习，分为两步：1）在训练集上训练word vectors W和paragraph vectors D，2）在推断阶段，固定W，训练D，得到测试集文本的paragraph vectors，再使用分类器以D为特征训练分类模型；

3、《Deep Unordered Composition Rivals Syntactic Methods for Text Classification》

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/17.png)

监督学习，使用文本里words的vectors的均值，使用更深的feed-forward network，及使用dropout（训练时随机漏掉文本中的某些word）。


