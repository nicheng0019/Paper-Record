《A Comparison of Event Models for Naive Bayes Text Classification》

设mixture model的components是c，参数为theta，documents是d，words是w，词汇表是V，则一个document的似然值为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/181.png)
 
假设每个mixture model component对应一个class；

(1)	Multi-variate Bernoulli Model

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/182.png)

B为0或1，表明单词w(t)是否出现在文本d(i)中，

使用训练样本来估计w在class c中的出现的概率：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/183.png)

P(c|d)为0或1，由document d的class决定，

class的先验参数，通过极大似然估计得到：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/184.png)

Multi-variate Bernoulli Model只考虑word是否出现，不考虑word出现的次数，没有出现的word也要把不出现的概率计算进去；

(2)	Multinomial Model

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/185.png)

N(it)是word t出现在document d中的次数，

w在class c中的出现的概率为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/186.png)

Class的先验概率和Multi-variate Bernoulli Model相同；

(3)	Classification

使用Bayes公式：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/187.png)

所以这种分类模型是generative model。

(4)	Feature Selection

选择average mutual information最高的words来作为特征。设C是所有classes的随机变量，W(t)取值0或1，表明word w(t)是否出现，H(C)是class变量的熵，H(C|W(t))是在word出现或不出现的条件下class变量的熵，则average mutual information值为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/188.png)

multi-variate Bernoulli model的计算：P(c)是所有class c的documents的个数除以documents的总数，P(f(t))是word w(t)出现的documents的个数除以documents的总数，P(c，f(t))是在class c的documents中出现word w(t)的个数除以documents的总数。

multinomial model的计算：P(c)是class c的documents中word出现的次数除以word出现的总次数，P(f(t))是word w(t)出现的次数除以word出现的总次数，P(c，f(t))是在class c的documents中word w(t)出现的次数除以word出现的总次数。

(5)	Conclusions

两种models都使用Bayes框架，区别在于计算P(d|c)的方式不同，multinomial model通常比multi-variate Bernoulli model的效果好。

注：

《Text Classification from Labeled and Unlabeled Documents using EM》里使用的是multinomial model，但计算P(d|c)的方法为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/189.png)


《Unsupervised Learning by Probabilistic Latent Semantic Analysis》 （《Probabilistic latent semantic indexing》）

(1) The Aspect Model

假设

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/230.png)

则联合概率模型为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/231.png)

似然函数为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/232.png)

改变D和Z的箭头方向的一个等价表示：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/233.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/234.png)

(2) Model fitting with the EM algorithm

E-step：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/235.png)

M-step：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/236.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/237.png)

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/238.png)

n(d)是document d的长度，即d中所有words的个数。



《Latent Dirichlet Allocation》

《Text Classification from Labeled and Unlabeled Documents using EM》中的模型为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/190.png)

P(w)为document的概率，z是topics，这个模型没有获取一个document表达多个topics的概率；

《Probabilistic latent semantic indexing》（《Unsupervised Learning by Probabilistic Latent Semantic Analysis》）中的模型为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/191.png)

或：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/192.png)

z为hidden topic，这个模型得到了document包括多个topics的概率，但是P(z|d)的值只能在训练集上得到；

LDA模型：

设一个document w为<w(1)…w(N)>，theta从Dirichlet(alpha(1)…alpha(k))分布中采样，topic z从以theta为概率的多项式分布中采样，在给定z的条件下，word w(n)被采样的概率为P(w(n)|z)，则document的概率为：

![image](https://github.com/nicheng0019/Paper-Record/blob/master/image/193.png)

Inference and learning：

使用《An Introduction to Variational Methods for Graphical Models》中的方法。


