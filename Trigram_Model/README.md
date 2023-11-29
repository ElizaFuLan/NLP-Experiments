## README

### Part1:

```python
def get_ngrams(sequence, n):
    # 定义一个函数，用于获取n-gram序列
    # 参数sequence是输入序列，参数n是n-gram的大小
    ngrams = []  # 存储n-gram的列表
    if n <= 0:
        return ngrams  # 如果n小于等于0，返回空列表
    
    # 添加START标记和STOP标记
    sequence = ['START'] * (n-1) + sequence + ['STOP']
    # print(len(sequence))
    
    for i in range(len(sequence) - n + 1):
        # 循环遍历输入序列，取连续的n个词语组成一个n-gram
        ngram = tuple(sequence[i:i + n])
        ngrams.append(ngram)
        # print(ngrams)

    return ngrams
```

当词组为1元、2元时，用一个‘START’、‘STOP’初始化即可。

当词组为3元时，需要用两个‘START’进行初始化。

### Part2:

```python
def count_ngrams(self, corpus):
        # 计算n-gram的函数
        # 参数corpus是语料库迭代器

        self.unigramcounts = {}  # 存储一元语法计数
        self.bigramcounts = {}   # 存储二元语法计数
        self.trigramcounts = {}  # 存储三元语法计数

        ##Your code here
        corpus_list = list(corpus)
        
        prev_word = "START"
        prev_prev_word = "START"

        for sentence in corpus_list:
            sentence = ["START"] * 2 + sentence + ["STOP"]
            for word in sentence:
                if word in self.unigramcounts:
                    self.unigramcounts[word] += 1
                else:
                    self.unigramcounts[word] = 1

                if prev_word:
                    bigram = (prev_word, word)
                    if bigram in self.bigramcounts:
                        self.bigramcounts[bigram] += 1
                    else:
                        self.bigramcounts[bigram] = 1
                
                if prev_prev_word:
                    trigram = (prev_prev_word, prev_word, word)
                    if trigram in self.trigramcounts:
                        self.trigramcounts[trigram] += 1
                    else:
                        self.trigramcounts[trigram] = 1
                prev_prev_word = prev_word
                prev_word = word
        # 此处应添加代码来计算n-gram的计数
        return
```

此处使用两个start进行初始化，以满足正确的三元词组数目输出。

### Part3:

```python
def raw_trigram_probability(self, trigram):
        # 计算原始（未平滑）三元语法概率的函数
        if trigram in self.trigramcounts:
            trigram_count = self.trigramcounts[trigram]
            bigram = trigram[:-1]  # Extract the bigram (first two words)
            if bigram in self.bigramcounts:
                bigram_count = self.bigramcounts[bigram]
                probability = trigram_count / bigram_count
                return probability
        return 0.0

    def raw_bigram_probability(self, bigram):
        # 计算原始（未平滑）二元语法概率的函数
        if bigram in self.bigramcounts:
            bigram_count = self.bigramcounts[bigram]
            unigram = bigram[:-1]  # Extract the unigram (first word)
            if unigram in self.unigramcounts:
                unigram_count = self.unigramcounts[unigram]
                probability = bigram_count / unigram_count
                return probability
        return 0.0
    
    def raw_unigram_probability(self, unigram):
        # 计算原始（未平滑）一元语法概率的函数

        # 提示：每次调用此方法重新计算分母可能会很慢！
        # 可以在TrigramModel实例中计算总词数，然后重复使用它。
        if unigram in self.unigramcounts:
            unigram_count = self.unigramcounts[unigram]
            total_words = sum(self.unigramcounts.values())  # Total number of words，使用了实例中的总词数
            probability = unigram_count / total_words
            return probability
        return 0.0
```

注意：预计许多n-gram概率为0，也就是说在训练中没有观察到n-gram序列的情况。您将遇到的一个问题是，当您有一个三元组u,w,v，其中count(u,w,v) = 0，但count(u,w)也为0时。在这种情况下，不清楚P(v | u,w)应该是什么。我的建议是，如果count(u,w)为0，则将P(v | u,w)设置为1 / |V|（其中|V|是词汇表的大小）。也就是说，如果三元组的上下文是未见的，则该上下文中所有可能单词的分布是均匀的。另一个选择是使用v的一元组概率，因此P(v | u,w) = P(v)。

添加拉普拉斯平滑后的函数如下：

```python
class TrigramModel(object):
    
    # ... 其他方法和构造函数 ...

    def raw_trigram_probability(self, trigram):
        """
        计算原始（未平滑）三元语法概率的函数，使用Laplace平滑
        """
        trigram_count = self.trigramcounts.get(trigram, 0)  # Get the trigram count or default to 0 if unseen
        bigram = trigram[:-1]  # Extract the bigram (first two words)
        bigram_count = self.bigramcounts.get(bigram, 0)  # Get the bigram count or default to 0 if unseen

        # Laplace smoothing
        vocabulary_size = len(self.lexicon)
        probability = (trigram_count + 1) / (bigram_count + vocabulary_size)

        return probability

    def raw_bigram_probability(self, bigram):
        """
        计算原始（未平滑）二元语法概率的函数，使用Laplace平滑
        """
        bigram_count = self.bigramcounts.get(bigram, 0)  # Get the bigram count or default to 0 if unseen
        unigram = bigram[:-1]  # Extract the unigram (first word)
        unigram_count = self.unigramcounts.get(unigram, 0)  # Get the unigram count or default to 0 if unseen

        # Laplace smoothing
        vocabulary_size = len(self.lexicon)
        probability = (bigram_count + 1) / (unigram_count + vocabulary_size)

        return probability
    
    def raw_unigram_probability(self, unigram):
        """
        计算原始（未平滑）一元语法概率的函数，使用Laplace平滑
        """
        unigram_count = self.unigramcounts.get(unigram, 0)  # Get the unigram count or default to 0 if unseen

        # Laplace smoothing
        vocabulary_size = len(self.lexicon)
        total_words = sum(self.unigramcounts.values())  # Total number of words
        probability = (unigram_count + 1) / (total_words + vocabulary_size)

        return probability

```

#### 生成句子：

```python
import random

class TrigramModel(object):
    
    # ... 其他方法和构造函数 ...

    def generate_sentence(self, t=20):
        """
        生成随机句子的函数
        参数t指定最大长度，但句子可能较短（如果遇到"STOP"标记）
        """
        sentence = []
        current_context = ("START", "START")  # Initialize with ("START", "START")

        for _ in range(t):
            # 获取当前上下文中所有可能的下一个词
            possible_next_words = []
            for word in self.lexicon:
                trigram = current_context + (word,)
                trigram_probability = self.raw_trigram_probability(trigram)
                possible_next_words.extend([word] * int(trigram_probability * 1000))  # Scale probabilities for random.choice

            # 随机选择下一个词
            next_word = random.choice(possible_next_words)

            # 将下一个词添加到句子中
            sentence.append(next_word)

            # 更新当前上下文
            current_context = (current_context[-1], next_word)

            # 如果生成了"STOP"标记，停止生成词
            if next_word == "STOP":
                break

        return sentence
```

这个函数首先初始化一个空的句子列表和当前上下文，然后进入循环，从当前上下文中获取可能的下一个词，并使用概率来随机选择下一个词。随后，它将下一个词添加到句子中，并更新当前上下文。如果生成了"STOP"标记，循环将停止，否则它将在达到最大长度（由参数t指定）时停止。

这里使用了一个简单的方法来缩放概率，以便在随机选择时更容易处理。这是因为`random.choice` 期望每个可能的下一个词具有相应的概率权重。因此，我们将每个可能的下一个词重复添加到列表中，以便其重复次数与其概率成正比例。

### Part4:

```python
def smoothed_trigram_probability(self, trigram):
        # 计算平滑后的三元语法概率的函数（使用线性插值）
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram_prob = self.raw_trigram_probability(trigram)

        # 获取对应的二元和一元概率
        bigram = trigram[:-1]
        bigram_prob = self.raw_bigram_probability(bigram)
        unigram = trigram[-1]
        unigram_prob = self.raw_unigram_probability(unigram)

        # 使用线性插值计算平滑后的概率
        smoothed_prob = lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * unigram_prob

        if smoothed_prob:
            return smoothed_prob
        else:
            return 0.0
```

$[:-1]$表示从第一个元素到倒数第二个元素进行切片

$[:]$表示从第一个元素开始切片

$[-1]$表示获取最后一个元素

### Part5:

```python
 def sentence_logprob(self, sentence):
        # 计算整个句子的对数概率的函数
        log_prob = 0.0
        trigrams = get_ngrams(sentence, 3)
        for trigram in trigrams:
            if self.smoothed_trigram_probability(trigram) > 0:
                trigram_log_prob = math.log2(self.smoothed_trigram_probability(trigram))
            else:
                trigram_log_prob = float("-inf")

            log_prob += trigram_log_prob
        return log_prob
```

### Part6:

