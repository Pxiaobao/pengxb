import nltk
import pdb
#nltk.download("brown")
from nltk.corpus import brown
import random
from tqdm import tqdm
import numpy as np
import math

corpus = brown.words()

from collections import Counter
'''
词汇大小为10000，为语料库中出现频率最大的前10000个单词创建一个词典来存储单词和ID的对应关系
'''
VOCAB_SIZE = 10000

counter = Counter(corpus)
vocab = counter.most_common(VOCAB_SIZE) #统计每个单词出现的次数zuiduode 10000个词[('the', 62713), (',', 58334), ('.', 49346), ('of', 36080), ('and', 27915), ('to', 25732)]
word2id = {w[0]:n for n,w in enumerate(vocab)}
#id2word
id2word = {n:w[0] for n,w in enumerate(vocab)}

'''
建立训练数据集，包含负取样产生的负数据集和从语料库中获得的正数据集。我们将窗口值设为2，负取样数量定为正样本的10倍
'''

WINDOW_SIZE = 2
NEG_SAMPLES_FACTORS = 10

train_set = list()
tokens = list(corpus)

for i, f_mid_token in tqdm(enumerate(tokens)):
  if f_mid_token in vocab:
    context_words = tokens[i+1, i+1+WINDOW_SIZE]
    f_mid_token_id = word2id[f_mid_token]
    for context_word in context_words:
      if context_word in vocab:
        context_word_id = word2id[context_word]
        train_set.append((f_mid_token_id, context_word_id, True))
        for i in range(NEG_SAMPLES_FACTORS):
          false_context_id = random.randint(0, VOCAB_SIZE-1)
          if false_context_id != context_word_id:
            train_set.append((f_mid_token_id, false_context_id, False))

  b_mid_token = corpus[i-VOCAB_SIZE]
  if b_mid_token in vocab:
    b_mid_token_id = word2id[b_mid_token_id]
    context_words = tokens[i-VOCAB_SIZE-WINDOW_SIZE: i-VOCAB_SIZE]
    for context_word in context_words:
      if context_word in vocab:
        context_word_id = word2id[context_word]
        train_set.append((b_mid_token_id, context_word_id, True))
        for i in range(NEG_SAMPLES_FACTORS):
          false_context_id = random.randint(0, VOCAB_SIZE-1)
          if false_context_id != context_word_id:
            train_set.append((f_mid_token_id, false_context_id, False))    

#初始化原始词向量，我们设置词向量维度为50

NUM_DIM = 50
target_word_matrix = 0.1 * np.random.randn(VOCAB_SIZE, NUM_DIM)
context_word_matrix = 0.1 * np.random.randn(VOCAB_SIZE, NUM_DIM)
pdb.set_trace()
#定义一个函数，用来更新初始化的词向量矩阵。在此之前我们先定义一个用来计算两个向量点积的Sigmoid函数值


def sigmoid(x):
  dot_prod = vec1.dot(vec2)
  return 1/(1+math.exp(-dot_prod))

def update(target_id, context_id, label, lr):
  context_vec = context_word_matrix[context_id]
  target_vec = target_word_matrix[target_id]
  prob = sigmoid(context_vec, target_vec)
  new_vec_ctxt = context_vec + lr * (label - prob) * target_vec
  new_vec_tgt = target_vec + lr * (label-prob) * context_vec
  # update the embedding matrix
  context_word_matrix[context_id] = new_vec_ctxt
  target_word_matrix[target_id] = new_vec_tgt

  log_likelihood = math.log(prob) if label else math.log(1-prob)
  return log_likelihood

#训练模型利用上面定义的更新函数以及通过负取样获取的训练集，我们对词向量矩阵进行优化。这里让训练集迭代2次，学习率设为0.1
EPOCH = 2
LR = 0.1

for _ in range(EPOCH):
  random.shuffle(train_set)
  log_likelihood = 0
  for tgt_id, ctxt_id, label in tqdm(train_set):
    log_likelihood += update(tgt_id, ctxt_id, label, LR)
  print("Log Likelihood: %.4f" % (log_likelihood))

#经过训练之后的**目标词的词向量矩阵**即为通过负取样方法获得的word2vec词向量。我们可以利用word2vec词向量寻找和给定单词最相似的n个单词
def most_similar_words(word, n):
  if word not in word2id:
    return []
  w_id = word2id[word]
  vec =target_word_matrix[w_id, :]
  m = target_word_matrix
  dot_m_v = m.dot(vec.T) # n-dim vector
  dot_m_m = np.sum(m * m, axis=1) # n-dim vector
  dot_v_v = vec.dot(vec.T) # float
  sims = dot_m_v / (math.sqrt(dot_v_v) * np.sqrt(dot_m_m))
  return [id2word[idx] for idx in (-sims).argsort()[:n]]  

print(most_similar_words('mother', 10))
