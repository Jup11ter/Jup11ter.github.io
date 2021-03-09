---
layout:     post
title:      Some experience about text preprocessing in NLP text classification and sentiment analysis task
subtitle:   A little experience of NLP preprocess
date:       2020-10-22
author:     Xiaoyu Zhang
header-img: img/post-bg-github-cup.jpg
catalog: true
tags:
    - NLP preprocess
    - Experience in work
---


[toc]
# 浅谈NLP 文本分类/情感分析 任务中的文本预处理工作
## 前言
之所以心血来潮想写这篇博客，是因为最近在关注NLP文本分类这类任务中的文本预处理工作，想总结一下自己的所学所想，老规矩，本博文记载**仅供备忘与参考**，不具备学术价值，本文默认使用python3编程（代码能力是屎山级别的，请谅解），默认文本为英文，代码主要使用Pytorch（博主老笨蛋了，之前一直执迷不悟用Keras，现在刚刚开始用torch，怎么说呢，挺香的 XD）

## NLP相关的文本预处理
NLP文本预处理一直是一个很受关注的问题，当下最常用的文本预处理工具当属nltk，功能统一，api也很简单，安装的话直接输入：
```shell
pip install nltk
python#进入python
import nltk
nltk.download()#下载需要的内容
```
一般来讲，最简单最常见的预处理就是把一整段文本分词化（Tokenize），对于一段文本（Sentence），可以直接调用nltk库功能将其分词化，返回结果为一个词表（word list）。
```python
import nltk# 为方便，任何import都只在所有代码块中出现一遍，以后的也同理
word_list=nltk.word_tokenize(sentence)
```
一般来讲在预处理数据的时候还会选择去除标点以及不需要的url等等内容，因此我在自己做实验的时候选择使用以下配置来作为基础的预处理方法。
```python
import string
import re

PUNCT_TO_REMOVE = string.punctuation
url_pattern = re.compile(r'https?://\S+|www\.\S+')
sentence=url_pattern.sub(r'', sentence)
#remove punc
sentence=sentence.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
tmp_word_list=nltk.word_tokenize(sentence)
word_list=[]
for word in tmp_word_list:    
    #lower       
    word=word.lower()
    word_list.append(word)
```

事实上，文本预处理的方法是非常多样的，根据下边代码块中的参考内容链接，你可以找到各种各样数十种有针对性或者泛用的预处理方法，有的是为了处理Twitter中的一些tag，有的是是为了对文本进行词根化，有的是为了将双重否定转换成肯定……总而言之，**一切预处理方法都是为了使得NLP任务更好地被执行，使得数据集更容易也更好地被训练。因此在我们针对NLP任务选择预处理方法时也应当注意选择合适的方法。**如果我们在一个新闻数据集中使用去除Twitter中tag的预处理方法进行处理的话只会浪费时间。
```python
# 参考链接
https://medium.com/sciforce/text-preprocessing-for-nlp-and-machine-learning-tasks-3e077aa4946e
https://towardsdatascience.com/all-you-need-to-know-about-text-preprocessing-for-nlp-and-machine-learning-bc1c5765ff67
https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79
https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing
https://www.kaggle.com/theoviel/improve-your-score-with-text-preprocessing-v2
https://medium.com/datadriveninvestor/data-cleaning-character-encoding-b4e0e9c65b2a
https://github.com/Deffro/text-preprocessing-techniques/blob/master/techniques.py
```
当然，很多预处理方法在常见的场合并不适用，例如文本中[表情处理](https://www.aclweb.org/anthology/W18-6231.pdf)在Reuters新闻分类以及IMDB情感分析等常用任务上就没有什么用处。

为此我总结了5个**我认为**常用的预处理方法在下面的代码中
```python
# 1. stem词根化
porter = nltk.stem.porter.PorterStemmer()
tmp_word_list=nltk.word_tokenize(sentence)
word_list=[]
for word in tmp_word_list:        
    word=porter.stem(word)
    word_list.append(word)

# 2. spell check拼写检查
# pip install pyspellchecker
from spellchecker import SpellChecker
spell=SpellChecker()
tmp_word_list=nltk.word_tokenize(sentence)
word_list=[]
for word in tmp_word_list:    
    #lower             
    misspelled_words = spell.unknown(word.split())
    if word in misspelled_words:
        word_list.append(spell.correction(word))
    else:
        word_list.append(word)

# 3. negation否定词替换
token=nltk.word_tokenize(token)
word_list=[]  
i, l = 0, len(token)
while i < l:
    word = token[i]
    if word == 'not' and i+1 < l:
        ant = replace(token[i+1])
        if ant:
            word_list.append(ant)
            i += 2
            continue
    word_list.append(word)
    i += 1

def replace(self,word, pos=None):
    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
    antonyms = set()
    for syn in nltk.corpus.wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None   

# 4. stop word 停用词替换
stops_list = set(nltk.corpus.stopwords.words('english'))
tmp_word_list=nltk.word_tokenize(token)
word_list=[]
for word in tmp_word_list:    
    if word not in stops_list:
        word_list.append(word)

# 5. contraction 连接词分离
# pip install contractions
import contractions as ctr
tmp_word_list=token.split(' ')
word_list=[]
for word in tmp_word_list:    
    word=ctr.fix(word)
    tmp=nltk.word_tokenize(word)
    for w in tmp:
        word_list.append(w)  

```

## 对BERT模型FineTune阶段数据集预处理效果分析
[BERT](https://arxiv.org/pdf/1810.04805.pdf)这类transformer预处理模型的特点是该类模型首先会在一个较大的语料库上进行训练，随后训练好的预处理模型在用户使用时只需要做一个简单的FineTune即可获得较好的效果。关于BERT网络的原理与分析，可以参考其他专业人士的[博客](https://blog.csdn.net/triplemeng/article/details/83053419?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param)详解，在此我不再赘述。

值得一提的是近期博主尝试了一下在BERT模型Finetune环节加入数据集的预处理，试图进一步改善模型的训练效果。博主尝试了一下在IMDB数据集上进行实验，实验使用5-fold交叉验证，严格控制了每个batch中的数据顺序，模型加载自transformer库的“bert-base-uncased”模型。预处理方法使用了上文提到的basic,stem,contraction,negation,stop五种方法的不同组合方式，包括‘basic+xx‘和‘all-xx’。其中“+”代表两种方法一同使用，“-”表示所有方法单独排除某种方法再一起使用。实验中记录最优val_loss与val_accuracy，并在5-fold交叉验证后进行平均，结果如下图所示。

| |loss|accuracy|
|--|--|--|
|no preprocess|**0.175**|**0.934**|
|basic|0.186|0.927|
|basic+stem|0.240|0.902|
|basic+contraction|0.182|0.933|
|basic+stop|0.227|0.907|
|basic+negation|0.194|0.931
|all|**0.532**|**0.654**|
|all-stem|0.223|0.916|
|all-stop|0.316|0.831|
|all-negation|0.605|0.581|
|all-contraction|0.257|0.894|
令我感到十分尴尬的是，所有预处理一起使用的效果非常之差，而加入预处理后的最好效果也只能说几乎和不做预处理持平……而在我做实验之前的认知里，参考关于ML模型的[预处理研究](https://www.sciencedirect.com/science/article/pii/S0957417418303683?casa_token=jo_i_0M7V7YAAAAA:eT8U_Qte4aYH30ZSB5djYmwJpNPDn7OCydgOynhFMzLlzKeGWJbpO-eYzPLD7-0pUcP6PlaNhZI)，预处理理应对语言模型的训练产生一定的正面影响——起码不应该是如此负面的效果……
在我与几个朋友讨论后，我们认为造成该现象的原因可能与模型的预训练相关，**BERT原始模型的预训练为保证学到上下文语义联系，数据集是未经过任何与处理的，而我在FineTune时加入预处理可能破坏了此时数据集的上下文文本关系，进而导致训练效果变差。**而对于ML模型本身未经过预训练，全靠模型在训练时自行学习上下文关系，因此合适的预处理会对训练效果带来不错的提升。

总结，个人经验来讲，对于BERT这种预训练模型，最经济实惠的方式还是直接在原始数据集加载预训练模型进行FineTune。

后续会再分享一些NLP预处理方面读论文想法与思考。