## README

### Part1:

```python
from math import isclose

def verify_grammar(self):
    """
    Return True if the grammar is a valid PCFG in CNF.
    Otherwise return False. 
    """
    # Checking CNF format for all rules
    for lhs, rules in self.lhs_to_rules.items():
        for rule in rules:
            _, rhs, _ = rule
            # If it's not a terminal (length is not 1)
            if len(rhs) != 1 and len(rhs) != 2:
                return False
            # If it's a binary rule, both should be non-terminals
            if len(rhs) == 2 and (rhs[0] in self.rhs_to_rules or rhs[1] in self.rhs_to_rules):
                return False

    # Ensuring probabilities sum up to 1 (or close to 1) for the same lhs
    for lhs, rules in self.lhs_to_rules.items():
        total_prob = fsum([prob for _, _, prob in rules])
        if not isclose(total_prob, 1.0, rel_tol=1e-9):  # assuming a very small tolerance for float inaccuracies
            return False

    return True
```

---

下面是`verify_grammar`函数的详细解释：

此函数旨在验证文法是否为Chomsky Normal Form (CNF) 中的有效Probabilistic Context-Free Grammar (PCFG)。在CNF中，每条规则或是连接两个非终结符，或是产生一个终结符。

1. `from math import isclose`：从`math`库导入`isclose`方法，用于后面的浮点数近似相等检查。

2. 对于`self.lhs_to_rules.items()`中的每个左侧符号及其规则，执行以下操作：

   - 对于每条规则，检查右侧是否有两个元素（非终结符）或一个元素（终结符）。
     - 如果`rhs`的长度不为1且不为2，返回`False`，因为它不符合CNF的要求。
     - 如果`rhs`有两个元素（这意味着它是一个二元规则），但它们中的任何一个不是非终结符（即不在`self.rhs_to_rules`字典中），则返回`False`。

3. 接着，再次遍历`self.lhs_to_rules.items()`：

   - 对于每个`lhs`及其相关规则，计算规则概率的总和。
   - 使用`isclose`方法检查这个总和是否接近1.0（允许一点小的误差，使用`rel_tol=1e-9`来定义这个小的误差）。如果不接近1.0，返回`False`。

4. 如果上述所有检查都通过，函数返回`True`，意味着文法是有效的CNF中的PCFG。

这个函数的主要目的是确保给定的文法满足PCFG的要求，并且其形式也符合CNF的要求。

---

### Part2:

```python
def is_in_language(self, tokens):
    """
    Membership checking. Parse the input tokens and return True if 
    the sentence is in the language described by the grammar. Otherwise
    return False
    """
    n = len(tokens)
    table = [[set() for _ in range(n+1)] for _ in range(n+1)]

    # Fill in the table for terminals
    for i in range(n):
        for lhs, rhs, _ in self.grammar.rhs_to_rules.get((tokens[i],), []):
            table[i][i+1].add(lhs)

    # Fill in the table for longer spans
    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for k in range(i+1, j):
                for rule_list in self.grammar.lhs_to_rules.values():
                    for rule in rule_list:
                        A = rule[0]
                        rhs = rule[1]
                        if len(rhs) == 2:
                            B, C = rhs
                            if B in table[i][k] and C in table[k][j]:
                                table[i][j].add(A)

    # Check if the start symbol can generate the whole sentence
    return self.grammar.startsymbol in table[0][n]
```

为了实现CKY算法，首先我们需要初始化一个解析表。此表的大小为`n x n`，其中`n`是句子的长度。对于每个单元`(i, j)`，我们需要查看文法中的所有规则，并检查是否可以从`i`到`j`应用这些规则。

以下是解析的步骤：

1. 初始化一个`n x n`的解析表，其中`n`是句子的长度。
2. 填写表中的对角线，即为长度为1的单元。这是通过查找能够生成每个终端的规则来完成的。
3. 使用CKY算法填写表的其余部分。对于表中的每个单元`(i, j)`，我们检查是否有规则`A -> B C`，使得`B`可以生成从`i`到`k`的子序列，且`C`可以生成从`k`到`j`的子序列，其中`k`是`i`和`j`之间的任意点。
4. 最后，我们检查是否可以从文法的开始符号生成整个句子，即检查开始符号是否在单元`(0, n)`中。

这样，如果句子可以由给定的文法解析，该函数就会返回`True`，否则返回`False`。

代码明确地处理了规则的结构，确保只在符合`(A, (B, C), _)`形式的规则上尝试解包。

---

### Part3:

```python
def parse_with_backpointers(self, tokens):
    """
    Parse the input tokens and return a parse table and a probability table.
    """
    n = len(tokens)
    table = defaultdict(dict)
    probs = defaultdict(dict)

    # Initialize tables with terminal rules
    for i in range(n):
        for lhs, rhs, prob in self.grammar.rhs_to_rules.get((tokens[i],), []):
            table[(i, i+1)][lhs] = rhs[0]
            probs[(i, i+1)][lhs] = math.log(prob)

    # Fill in the tables for longer spans
    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for k in range(i+1, j):
                for rule_list in self.grammar.lhs_to_rules.values():
                    for rule in rule_list:
                        A = rule[0]
                        rhs = rule[1]
                        prob = rule[2]
                        if len(rhs) == 2:
                            B, C = rhs
                            if B in table[(i, k)] and C in table[(k, j)]:
                                new_prob = math.log(prob) + probs[(i, k)][B] + probs[(k, j)][C]
                                if A not in probs[(i, j)] or new_prob > probs[(i, j)][A]:
                                    table[(i, j)][A] = ((B, i, k), (C, k, j))
                                    probs[(i, j)][A] = new_prob

    return table, probs
```

此方法首先为单词初始化表，使用文法中的终结符规则。然后，它使用CKY算法的递归结构填充更长的范围。对于每个可能的分裂和规则组合，它检查新的分裂是否会产生更高的对数概率，并相应地更新表格。

在这个实现中，我们使用了`defaultdict`，这是Python的一个方便的数据结构，它可以在访问不存在的键时返回一个默认值。这使得代码更简洁，因为我们不需要首先检查键是否存在。

---

### Problem4:

问题：执行该函数时，evaluate后的coverage总是100%：

```python
def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if nt not in chart[(i, j)]:  
        return None
    
    rule = chart[(i, j)][nt]

    if isinstance(rule, str):
        return (nt, rule)
    
    left_nt, left_i, left_j = rule[0]
    right_nt, right_i, right_j = rule[1]

    left_tree = get_tree(chart, left_i, left_j, left_nt)
    right_tree = get_tree(chart, right_i, right_j, right_nt)

    return (nt, left_tree, right_tree)
```

这个函数构建并返回基于非终端`nt`和跨度`i,j`的解析树。它采用以下步骤：

1. 检查`chart`中是否存在给定的非终端和跨度组合。如果不存在，则返回`None`。
2. 根据跨度和非终端从`chart`中提取规则。
3. 如果规则是一个字符串（即它是一个终端符号），那么直接返回该非终端与其关联的终端的组合。
4. 否则，它递归地调用`get_tree`函数，为左侧和右侧的子树分别构建解析树。
5. 最后，它返回当前非终端与左侧和右侧子树的组合。

现在，关于Coverage为100%的原因：

- Coverage基于是否为每个句子生成了一个解析。从`get_tree`函数中，我们可以看到，除非给定的跨度和非终端在`chart`中不存在，否则它始终返回一个解析树。当然，这不意味着每个解析都是正确的，但它确实为每个句子生成了一个解析。
- `get_tree`不检查是否生成的树是语法上的最佳解析或是否它对应于最高的概率。它简单地根据提供的信息生成一个树。
- 对于每个给定的跨度和非终端，`parse_with_backpointers`函数在`chart`中存储了一个规则，这意味着对于每个给定的跨度和非终端组合，`get_tree`函数总是有一些东西可以返回，即使它不是最佳的或最准确的解析。

总之，由于`get_tree`的构造方式，它总是尽量返回一个解析树，这可能导致Coverage达到100%。但这并不意味着每个生成的解析都是正确的或最佳的。

注意，在本任务中，要求使用递归遍历生成树，最终解决为如下代码：

```python
def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if j - i == 1:
        output = (nt, chart[i, j][nt])
        return output

    out1 = get_tree(chart, chart[(i, j)][nt][0][1], chart[(i, j)][nt][0][2], chart[(i, j)][nt][0][0])
    out2 = get_tree(chart, chart[(i, j)][nt][1][1], chart[(i, j)][nt][1][2], chart[(i, j)][nt][1][0])
    return (nt, out1, out2)
```

#### 关于本代码的解释：

```python
def get_tree(chart, i, j, nt): 
```
定义一个名为`get_tree`的函数，该函数接收四个参数：`chart`（一个保存解析信息的字典）、`i`（起始位置）、`j`（结束位置）以及`nt`（一个非终端符号）。

```python
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
```
这是函数的文档字符串，描述了函数的主要功能：返回以非终端符号`nt`为根、覆盖从`i`到`j`的范围的解析树。

```python
    # TODO: Part 4
```
这是一个注释，可能是提示开发者完成第4部分的任务。

```python
    if j - i == 1:
```
检查`j`和`i`之间的距离是否为1。如果为1，这意味着这段范围只包含一个单词。

```python
        output = (nt, chart[i, j][nt])
```
从`chart`中获取与此非终端符号`nt`对应的词，并与`nt`一起创建一个元组。

```python
        return output
```
返回上述创建的元组。

```python
    out1 = get_tree(chart, chart[(i, j)][nt][0][1], chart[(i, j)][nt][0][2], chart[(i, j)][nt][0][0])
```
递归地为左子树调用`get_tree`函数，获取与当前非终端符号`nt`对应的第一个子节点的解析树。

```python
    out2 = get_tree(chart, chart[(i, j)][nt][1][1], chart[(i, j)][nt][1][2], chart[(i, j)][nt][1][0])
```
递归地为右子树调用`get_tree`函数，获取与当前非终端符号`nt`对应的第二个子节点的解析树。

```python
    return (nt, out1, out2)
```
将当前的非终端符号`nt`与其两个子节点（左子树和右子树）组合成一个元组，并返回。

这个函数基于递归，目的是构建并返回一个给定非终端符号`nt`的解析树，覆盖从`i`到`j`的范围。