## homework3

#### part1

第 1 部分 - 获取词汇 (0 分)
因为我们将对单词和 POS 标签使用 one-hot 表示，所以我们需要知道哪些单词出现在数据中，并且需要从单词到索引的映射。

运行以下命令

$python get_vocab.py 数据/train.conll 数据/words.vocab 数据/pos.vocab
生成单词索引和 POS 索引。 这包含在训练数据中出现多次的所有单词。 单词文件将如下所示：

<CD> 0
<NNP> 1
<未知>2
<根> 3
<空> 4
封锁5
飓风 6
船舶 7
前 5 个条目是特殊符号。 <CD> 代表任何数字（任何带有 POS 标签 CD 标签的内容），<NNP> 代表任何专有名称（任何带有 POS 标签 NNP 标签的内容）。 代表未知的单词（在训练数据中，任何仅出现一次的单词）。 <ROOT> 是一个特殊的根符号（与单词 0 关联的单词，最初放置在依存解析器的堆栈上）。 <NULL> 用于填充上下文窗口。

#### part2

第二部分 - 提取用于训练的输入/输出矩阵（35分）
为了训练神经网络，我们首先需要获得一组输入/输出训练对。更具体地说，每个训练示例应该是一个对（x,y），其中x是一个解析器状态，y是解析器在该状态下应该进行的转换。

请查看文件`extract_training_data.py`
状态：输入将是类`State`的一个实例，该类代表一个解析器状态。这个类的属性包括一个堆栈、缓冲区和部分构建的依存结构deps。堆栈和缓冲区是单词id（整数）的列表。
堆栈的顶部是列表中的最后一个单词`stack[-1]`。缓冲区中的下一个单词也是列表中的最后一个单词`buffer[-1]`。
Deps是一系列（parent, child, relation）三元组，其中parent和child是整数id，relation是一个字符串（依赖标签）。

转换：输出是一对（transition, label），其中transition可以是"shift"、"left_arc"或"right_arc"中的一个，label是依赖标签。如果转换是"shift"，依赖标签是None。由于有45种依赖关系（见列表deps_relations），因此可能有45*2+1种输出。

获取神谕转换和输入/输出示例序列。
如我们在课上讨论的，我们不能直接从语料库观察到转换。我们只能看到结果依存结构。因此，我们需要将树转换成我们用于训练的（状态，转换）对序列。这部分已经在函数`get_training_instances(dep_structure)`中实现了。给定一个`DependencyStructure`实例，此方法将返回一个上述格式的（State, Transition）对列表。

待办事项：提取输入表示
你的任务将是将输入/输出对转换成适合神经网络的表示。你将完成类`FeatureExtractor`中的方法`get_input_representation(self, words, pos, state)`。`FeatureExtractor`类的构造函数将两个词汇文件作为输入（文件对象）。然后它在属性`word_vocab`中存储单词到索引的字典，在属性`pos_vocab`中存储POS到索引的字典。

`get_input_representation(self, words, pos, state)`将输入句子中的单词列表、输入句子中的POS标签列表和`State`类的一个实例作为参数。它应该返回一个编码到神经网络的输入，即一个单一向量。

为了表示一个状态，我们将使用缓冲区上的前三个单词和堆栈上的后三个单词，即`stack[-1]`、`stack[-2]`、`stack[-3]`和`buffer[-1]`、`buffer[-2]`、`buffer[-3]`。我们可以为每个单词使用嵌入表示，但我们希望网络自己学习这些表示。因此，神经网络将包含一个嵌入层，单词将被表示为一个独热表示。实际的输入将是每个单词的独热向量的连接。

这通常需要一个6x|V|的向量，但幸运的是keras嵌入层将接受整数索引作为输入并在内部转换它们。因此，我们只需要返回一个长度为6的向量（一个一维numpy数组）。

所以例如，如果缓冲区上的下一个单词是"dog eats a"，而"the"和<ROOT>在堆栈上，返回值应该是一个numpy数组`numpy.array([4047, 3, 4, 8346, 8995, 14774])`。这里4是<NULL>符号的索引，3是<ROOT>符号的索引，4047是"the"的索引，而8346、8995、14774是"dog"、"eats"和"a"的索引。（你的索引可能与这个示例不同，因为`get_vocab`脚本每次运行时输出的索引映射都不同。）

注意，在创建输入表示时，你需要考虑到特殊符号（<CD>、<NNP>、<UNK>、<ROOT>、<NULL>）。确保你考虑到堆栈或缓冲区上少于3个单词的状态。

这种表示是Chen & Manning（2014）论文中特征的一个子集。一旦你运行了基本版本，可以自由地尝试完整的特征集。

待办事项：生成输入和输出矩阵

编写方法`get_output_representation(self, output_pair)`，它应该将一个（transition, label）对作为其参数，并返回这些动作的独热表示。因为有45*2+1 = 91种可能的输出，输出应该表示为长度为91的独热向量。

保存训练矩阵
神经网络将接受两个矩阵作为输入，一个是训练数据矩阵（在基本情况下是N x 6矩阵，其中N是训练实例的数量），一个是输出矩阵（一个Nx91矩阵）。

函数`get_training_matrices(extractor, in_file)`将取一个`FeatureExtractor`实例和一个文件对象（一个CoNLL格式的文件）作为输入。然后它将提取状态-转换序列，并对每一个调用你的输入和输出表示方法来获得输入和输出向量。最后它将组装矩阵并返回它们。

`extract_training_data.py`中的主程序调用`get_training_matrices`来获取矩阵，然后将它们写入两个二进制文件（以numpy数组二进制格式编码）。你可以像这样调用它：

```
python extract_training_data.py data/train.conll data/input_train.npy data/target_train.npy
```

你也可以为开发集获取矩阵，这有助于调整网络参数。

```
python extract_training_data.py data/dev.conll data/input_dev.npy data/target_dev.npy
```