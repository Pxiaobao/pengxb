GPT-3是一个基于Transformer的[大型语言模型](https://www.zhihu.com/search?q=%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3079118257%7D)，它只使用了Transformer中的Decoder部分，没有使用Encoder部分。因此，ChatGPT也是一个只用了Decoder的模型，没有Encoder-Decoder的结构。

Decoder是一个由多个层组成的模块，每一层包含三个子层：[自注意力](https://www.zhihu.com/search?q=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3079118257%7D)（Self-Attention）、[交叉注意力](https://www.zhihu.com/search?q=%E4%BA%A4%E5%8F%89%E6%B3%A8%E6%84%8F%E5%8A%9B&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3079118257%7D)（Cross-Attention）和[前馈神经网络](https://www.zhihu.com/search?q=%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3079118257%7D)（Feed Forward Neural Network）。自注意力是用来处理输入序列的信息，交叉注意力是用来处理来自Encoder的编码信息（如果有的话），前馈神经网络是用来增加模型的非线性能力。在每个子层之后，还有一个[残差连接](https://www.zhihu.com/search?q=%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3079118257%7D)（Residual Connection）和一个层归一化（Layer Normalization）操作。

ChatGPT作为一个对话模型，它的输入序列是由用户输入和历史对话组成的，它的输出序列是由模型生成的回复组成的。因为它没有Encoder部分，所以它不需要交叉注意力子层，只需要自注意力和[前馈神经网络子层](https://www.zhihu.com/search?q=%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AD%90%E5%B1%82&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22answer%22%2C%22sourceId%22%3A3079118257%7D)。在自注意力子层中，它还使用了掩码（Mask）操作，来避免模型看到未来的信息。





### **seq2seq学习范式**

---

**Encoder-Decoder** 在正式介绍ilya的seq2seq范式之前，先接介绍下另一位大神yoshua的Encoder-Decoder模型：

•paper：learning phrase representations using RNN Encoder-Decoder for statistical machine translation•要点：•encoder：将变长sequence编码为一个fixed-width向量•decoder：将fixed-width向量解码为另一个变长sequence•模型：•p（f｜e）=p(e|f)*p(f)/p(e)；p（f）为语言模型；p（e｜f）翻译模型

该模型的出现主要是为了解决变长输入输出的问题，和输入输出尺寸比较固定的CV领域不同，在NLP领域，经常会碰到输入和输出的长度变化较大的任务类型；比如翻译、问答；每个具体sample的输入和输出可能都不相同，而一般的DNN模型对变长输入输出的处理只能通过padding（不足的补0之类）的方法来强行对齐到固定长度，这种做法一方面可能引入噪音，另一方面对不同元素之间的相互关系容易丢失 Encoder-Decoder的思路是将变长输入通过RNN结构编码为一个定长向量，然后解码阶段将定长向量、`h_{t-1}`、上一个输出item `y_{t-1}`一起作为输入，生成下一个输出item `y_t`，如图所示

![](https://pic4.zhimg.com/v2-c63438ce265241b337bfaa732288d8d7_b.jpg "null")

**seq2seq** 在Encoder-Decoder的基础上，ilya在《sequence to sequence learning with neural networks》提出了seq2seq模型，如图

![](https://pic3.zhimg.com/v2-21b1cfa9773a850346457f39389bc93e_b.jpg "null")

这篇文章从结构上看和Encoder—Decoder几乎是一模一样（实际上也没看出啥区别），主要的贡献点有：

1.用lstm替换之前的RNN，比较好的解决了短期记忆问题2.input序列采用逆序输入，效果大幅度提升3.将基于Decoder的Next token prediction问题和NLP一般任务做了些对比，初步形成NTP的思路

其中1和2的新颖度有限，虽然时候看1可能暗示了transformer自注意力的强大潜力；最重要的是将原本机器翻译领域的Encoder-Decoder模式以一种seq2seq的形式抽象出来，并将翻译、问答等NLP问题视作一种sequence到sequence的映射，因此，一种领域无关的sequence到sequence映射模型具有一般意义，这里虽然用到了**Next token prediction**的模式，还没有完全独立出来

### **transformer**

---

在之前的seq2seq范式里用到的模型组件一般是lstm、gru、rnn之类的，这些都属于RNN范畴的组件有一个共同特性：隐藏状态`h_t`依赖于上一个隐藏状态`h_{t-1}`，这种序列行为是RNN能抓取序列不同位置item相关关系的关键，但严重阻碍了模型训练的并行度 要解决这个问题，只能彻底去掉RNN类的结构，不再进行序列化的隐态`h_t`计算才有可能，但是RNN的隐态是捕获元素之间关系的结构，要去掉必须用其他东西来代替，而注意力（attention）机制恰好可以提供这个能力（注意力机制的详细论述不在本文范围，简单理解为一种加权融合就好）

![](https://pic4.zhimg.com/v2-e1b56bba937a837621ab9d6aef9f8ed3_b.jpg "null")

transformer的整体结构如图所示，由encoder和decoder模块构成；两个模块里最核心的部分是multi-head attention部分，其他组件都是基础NN结构；multi-head attention如下图所示

![](https://pic4.zhimg.com/v2-9c744cd85e177e067b2f69e559fdb823_b.jpg "null")

这里的attention用的是内积attention的变体（scaled解决振幅问题），简单理解就是根据query和一系列key的相似度情况（内积），用key对应的v加权和相似度进行加权得出Q的新的向量表达 multi-head是借鉴了CNN的思路，通过将QK映射到不同的线性子空间，再进行attention，就可以获得同一个序列的不同item再不同子空间的表达之间的相关关系 可以看到，用attention替换RNN的隐态后，不再需要序列化计算隐态`h_t`了，而是可以并行计算multi-head attention来获得item和context之间的相关关系；而后者是可以并行计算的，这使得基于大数据量进行LLM训练成为可能 剩下的两个问题：**自注意力的自是什么意思？没有RNN之后的item的位置信息如何表达？** 这两个也简单说下，自注意力的意思是QKV都是输入的字符串，这样就可以计算一个输入sequence内部各个token之间的关系了；因为是自己和自己进行attention，所以叫自注意力（self-attention）；位置信息需要通过额外的position embeding来表达

**token和tokenizer说明** 这里也简单说下token的含义，早期用分词器对文本进行分词，token和英文的word基本同义，但这种方法对于不在训练集里的word非常不友好，后来BPE（2015年左右更新，起源于1994年，参考资料里有相关材料）将单词拆分为更细粒度的token，如此一来，token和word不再严格一一对应，统计上一个token约等于0.75个word，BPE分词器也称为GPT系列标配

### **GPT-series**

---

### **GPT-1**

GPT-1的论文《Improving language understanding by generative pre-training》，这篇论文是在google的BERT之前，但是关注度远不如后者 在GPT-1之前，NLP任务一般采用一些词训练模型，比如skip-gram、CBOW之类的模型从自然文本里学习词向量后，再用特定任务的NN结构来做；GPT-1的思路是相对于之前的word embeding、phrase-embeding之类的做法，是否可以在海量的自然文本里学习到更高阶的语法语义知识，再用简单的任务相关的目标函数来微调模型以达到**任务无关的预训练模型**+微调解决NLP问题的目的 主要的贡献如下：

•GPT采用了Decoder-only的结构；这一思路借鉴了《generating wikipedia by summarizing long sequence》•采用了pretrain+finetuning模式，但我之前word embeding+task specific architecture不同•去掉了task-specific的模型结构，只有pretrained模型+线性层•模型大小130M左右，12层

模型的结构如图所示

![](https://pic1.zhimg.com/v2-56f8775bd66902e662ff1c3ec6ea4d90_b.jpg "null")

可以看到和标准的transformer不同，GPT-1仅用了Decoder模块，并且去掉了模块里与encoder有关的一轮multi-head attention，只保留了Masked Multi-head attention和FNN层以及正则层 因为GPT-2相对于1的模型结构变化不大，这里可以用开源的GPT-2的代码来印证下模型结构，如下图

![](https://pic4.zhimg.com/v2-7710b19401308dbaf8aa66be4ab3cf6f_b.jpg "null")

看到这里有个很基础的问题，去掉了encoder，如何对模型进行训练？借用《state of GPT》里的一张图来说明

![](https://pic2.zhimg.com/v2-1ceacec9e4ec74b006b3daae56d69dd1_b.jpg "null")

上图每个数字代表一个token，绿色为当前token，黄色为context，红色为要预测的目标token；通过对自然文本进行采样，自然能得到需要的序列，然后用LM的条件概率损失函数即可对模型进行训练和更新 这里需要指出的是在GPT-1的时候OpenAI就确定了**NTP（next token prediction）** 的技术路线；后续的2和3也都是沿着这条路线没有变更 相对于BERT的利用encoder-only结构，抓取双向上下文信息的做法，NTP路线在起步阶段，在很多的NLP标准任务上都是弱势；感兴趣的可以参考GPT-1之后出来的BERT的论文

### **GPT-2**

GPT-2的模型结构在上一节已经通过代码给出，和GPT-1比几乎没什么变动；在GPT-1出来之后的BERT在很多标准NLP任务上效果更好，这时候可以通过更大的数据集来训练一个更大的模型GPT-2，但是略有尴尬的是GPT-2在标准NLP任务上依然不如BERT，所以采取了一些取巧的做法，从标题《language models are unsupervised multitask learners》可以看出，GPT-2在去掉任务相关的结构方面更进一步：**彻底去掉任务相关的NN结构**，主要贡献如下：

•idea：一个足够大的语言模型，可以从输入的sequence里理解任务，并尝试完成任务；因此，不需要再叠加任何task-specific的结构；因此不需要supervised fine-tuning。•模型结构：和GPT-1类似，参数规模1.5B，context从512扩展到1024，layer从12扩大到48层•观念转变：multi-task学习并不需要显示的监督学习，而可以通过类似 **（translate engilist to french，english text，french text）** 的方式来表达，将原来需要通过监督学习来finetuning的任务转为了seq2seq的任务，从而让有监督的目标函数和无监督的目标函数一致•之前的bert在NLP标准任务上效果较好，但不是NTP模式；通过prompt可以将标准NLP任务转换为NTP形式•conclusion：在海量文本上训练的高参数容量的语言模型无需显示监督学习也可以较好的完成各种任务

GPT-2可以看到模型结构并没有什么变化，用了更大的数据集，最大的贡献就是尝试用自然语言来将标准NLP任务转换为NTP形式的任务，并且证明了随着模型参数的提升，模型在NTP类任务上的表现可以得到持续的改进

### **GPT-3**

•背景：pretrained model+finetuning存在一些问题•finetuning对特定任务的样本数的需求一般需要几千到几万个，大大限制了这个范式的应用范围•finetuing将模型在特定任务训练集合上微调，使模型能力变窄，可能损失一部分泛化性•人类并不需要一个较大的数据集才能学会某项任务

GPT-3《language models are few-shot learners》首先分析了早期给予预训练模型和finetuning方式存在的一些问题，结合GPT-2的结果，进一步抽象了通过自然语言来描述任务，引导语言模型通过NTP产生输出序列来完成任务的概念，定义为In-Context learnging（原文还有一个meta-learning的重定义，可以忽略），如图所示

![](https://pic4.zhimg.com/v2-8e635572353941ab139248b7691906a3_b.jpg "null")

虽然叫做In—Context Learning，但是并不会更新模型的参数；仅仅是通过编写输入序列（prompt）来给模型提供任务有关的信息，这样ICL和一般的SFT（会更新模型参数的supervised finetuning）就区分开；具体效果如图

![](https://pic2.zhimg.com/v2-a54dcb427f662bbb6a72a923298fb57d_b.jpg "null")

上图的横坐标为context里给出的例子数，纵坐标为准确率，不同颜色代表不同参数规模的模型，虚线和实现代表example之外的prompt设定，zero-shot、one-shot、few-shot代表输入里example的个数，从图可以看出：

•同等参数规模的模型，输入的例子数量越多效果越好•模型参数越大，ICL的效果越好 在更广泛的NLP任务测试同样保持了这个结果

![](https://pic4.zhimg.com/v2-7d6b01d33a32676c863ff07cb23d7e8b_b.jpg "null")

途中实线代表在多个任务上取均值，依然遵守上述规律 在之前的工程篇里提到过缩放定律，在GPT-3依然遵守，如图

![](https://pic1.zhimg.com/v2-69debfc20c87cff594af02bac9e5a598_b.jpg "null")

模型的效果随着算力投入和参数规模提升持续下降，结合缩放定律和GPT-3的ICL可以看出，LLM的效果还没到极限

### **instructGPT**

InstructGPT主要解决对齐（alignment）的问题

•Paper link:https://arxiv.org/abs/2203.02155

**为什么要对齐？**

![](https://pic2.zhimg.com/v2-39ff22f381b365306aa67c17a72eb7e9_b.jpg "null")

andrej在演讲里提到：模型更像一个token模拟器，只是产生一个序列来完成任务；利用这一点，可以用prompt来诱使基础模型称为一个AI助手 之所以这样，可能是因为：**LLM一般是基于网络预料训练的，和"遵从用户指令以产生安全而有用的结果"的目标并不一致，因此需要使用对齐（alignment）技术来将LLM和人类的意图以对齐**

具体如何对齐呢？要点如下：

•对齐目标：helpful（有用）、honest（改为truthful，遵从事实的）、harmless（无害的）•步骤：SFT--->RM--->PPO•收集一般性的数据，对LLM（GPT3）进行SFT•收集比较数据集（对不同结果排序），训练RM模型•使用RM模型，在RL环境用PPO训练以学习一个策略•步骤2和3可以反复迭代，流程如图所示 下图源于InstructGPT的paper

![](https://pic4.zhimg.com/v2-49371396f796104754180a53acd95e07_b.jpg "null")

从图可以看到，GPT-3某种程度上背离了不用SFT的做法，但是也紧紧是针对NTP来做SFT，通过人工标注一些优质结果，来引导模型产生更符合人类习惯的输出，在对齐的过程中，用了一个RM（奖励模型）和RLHF的方法，一方面是人工制作用来做SFT的样本数不够，另一方面发现模型在RLHF的情况下确实对齐效率更高；Andrej给出的一个可能解释是：或许做分别比做生成难，就像俗话说，没吃过猪肉还没见过猪跑 但这里复杂的PPO-RL是否有必要，如果有足够的样本，是否直接SFT也可以达到同样的效果恐怕要继续观察 另一个发现是**对齐税**的问题，在对齐的过程中，模型在更大范围的泛化性会有所丢失，在标注NLP任务上可能退化，论文里提到在做RLHF的时候混合一部分初始样本能缓解对齐税的问题

**或许要让LLM适应人类的表达方式需要付出额外的代价～** 针对GPT-3的其他一些结论摘记：

•模型对齐的成本相对于预训练成本非常小，SFT：PPO：Pretrain=4.9:60:3490•遵从指令的能力有很好的泛化性，甚至对于SFT没有涉及的任务类型也有效，原因不太清楚•标准NLPtask不能很好的反应LLM的能力，但依然要减少对齐税•对齐会导致模型在标准NLP任务效果下降，成为对齐税，RLHF里加入部分预训练样本可以大幅减轻对齐税•如果对一个问题的回复大家看法分歧较大，取均值并不是个可取的方法

### **ChatGPT**

ChatGPT并没有paper放出来，仅有**blog[3]**介绍，从介绍看，用到的技术和InstructGPT并没有区别，主要是数据集方面的差异，用andrej的图片来说明如下：

![](https://pic1.zhimg.com/v2-2e2b04b0cd4230d6ec01758e3a45b878_b.jpg "null")

这是InstructGPT里那张图的细化版本，里面可以看到明确的模型路径，比如RM是用SFT模型初始化的（并不是175B那个，而是一个小的多的SFT模型即可）

## **LLM SFT**

---

这一节简单介绍下PEFT的思路，论文还没细读，后续再补

•一般的SFT 一般的SFT还是遵循pretrained LM+finetuning的范式，在特定任务上准备一个数据集，然后使用相对小的学习率来更新pretrained LM（几乎会更新大模型的所有参数），以追求特定任务上的效果；这个方法的缺点是，如果有多个特定任务，每个任务都需要通过一个数据集来对初始基础模型做微调，每个任务上都有一个微调过的pretrained LM（一般都比较大），比较低效•adaptor：parameter-efficient transfer learning for NLP•LoRA•QLoRA

adapter的思路和一般的SFT不同，保持pretrained LM不变，针对每个任务额外训练一个非常小的参数模块，这个额外引入的、task-specific的模块被称为adaptor，这样在多个任务的情况下，进需要一个基础模型+一系列adaptor即可；adapter的大小可能不到pretrained LM的1%

![](https://pic1.zhimg.com/v2-fd51be15ae6fb67959a278240412122c_b.jpg "null")

通过LLM+adapter的方式，可以在多个特定任务上共享同一个基础模型，同时adapter也可能通过hub的方式发布，是一种比普通SFT更高效的方法

## **应用**

---

### **使用LLM改进业务的思路**

从LLM的演进来看，未来很长一段时间内，基础模型的训练都是拥有巨大算力基础设施的机构垄断的领域（即便云能提供能力，但是高昂的训练费用和RLHF的不稳定性依然会形成很高的门槛）； 此外，基础模型训练完成后，还需要对齐才能成为有用的AI助手，而庞大的模型规模在推理时的成本依然难以接受（一次问答如果需要几十张GPU来协同推理，成本是不可接受的），因此对齐后的模型真正变成可以广泛使用的服务还需要进一步对模型体量做压缩，因此大模型的蒸馏技术有了发挥的空间 最后，经过对齐和蒸馏的模型可以称为很好的AI助手，但缺乏私域数据，如何将领域独有数据和助手模型结合是短期应用层最活跃的领域，大致有：检索增强生成和SFT两个方向，前者将领域数据用检索设施汇集后，生产一些prompt，然后交给助手模型来生成结果；后者用领域数据直接对模型进行SFT，相当于将助手模型进一步窄化到具体领域，虽然会失去一些泛化性，但只要在特定领域的表现令人满意，也是一种价值 按照这个理解画了一个模型的应用层次的关系图如下：

![](https://pic4.zhimg.com/v2-e6be771468455c51e4f63a16a2dd6dc7_b.jpg "null")

在《stat of GPT》的演讲里，主持人给出的使用建议的顺序如图所示

![](https://pic2.zhimg.com/v2-22ac34658bc6e427849a2afefc85e471_b.jpg "null")

作者建议的使用LLM来解决具体业务需求的方法是两步走：

•尽量达到目标的最好效果•使用GPT-4•使用包含任务上下文、相关信息、指令的prompt•检索并添加相关上下文和信息到prompt•实验prompt工程技术•实验few-shot样例技术，尽量：a）和测试用例相关 b）有一定多样性•实验使用tool或者plugin将某些不适合LLM的任务卸载交给对应的工具•花时间优化pipeline/chain•如果你认为prompting的潜力最大化了，考虑收集SFT数据集+finetuning•专家领域：考虑RM data收集，RLHF finetuning•优化成本•如果已经拿到最好的效果，尝试节约费用的方法•使用GPT-3.5、使用更短的prompt等

### **职业考虑**

随着LLM的发展，可能的新的机会可能有：

•prompt工程师 Andrej在《state of GPT》里给出了自己对语言模型的看法，他认为LLM是一个庞大的token模拟器（simulator），模型实际并不了解具体token的含义，而是试图以自己的理解以NTP的方式去完成一个结果sequence，因此幻觉等问题持续存在。从这个角度，精心设计的prompt以及CoT等思路的，可以更好的去和LLM合作，激发出模型里有用的信息。相对于系统1，LLM更像是系统2，需要对复杂任务进行分解，人机协作，才能更好的去使用这个模拟器。因此，prompt工程师未来确实有可能成为一个职业，类似程序员写代码，写prompt来完成工作。一个风险点是：模型升级后，prompt的效果是否能保持以及prompt的需求会不会萎缩•模型训练 掌握算力infra构建能力，或者是基础模型训练能力的工程师是赛道里的顶级玩家了，注定是少数人；从推荐技术推广开始，AI含量越高的技术，对人数的绝对需求都是下降的，因此模型训练领域虽然是顶尖岗位，但是市场规模可能不会太大，毕竟在大力出奇迹的NTP路线里，需要的搬砖工不多•标注人员 模型的效果取决于训练数据的质量，这一规律在LLM仍然遵守；从GPT系列论文里数据集方面的工作介绍就可以看出来；在SFT阶段需要人工对不同的结果做比较排序，排序的水平同样会限制模型的最终效果；因此，高质量的数据标注也会制造一些岗位；包括类似的比较标注、也包括人工编写demonstration•SFT工程师 基于开源或商业模型，针对特定领域收集SFT数据集，对模型进一步SFT的需求也会存在，可能是领域专家收集数据集再配合SFT工程师来做，也可能有专门以SFT针对一些公开领域做些微调；除了基础模型，后续adaptor也可能成为一种可交易的模型资产，等于基础模型加上不同的adator外挂来更好的解决特定问题 除了新的职业，哪些职业更容易被淘汰呢？•初级程序员•初级内容创作者•不会与AI助手协作的人 考虑到模型已经具备了短链推理能力、few-shot学习能力，大量逻辑链条偏短的初级职位可能都会被挤出，领域专家有一定的壁垒，如果掌握与AI助手的协作能力，可能会延长一些职业生涯 可能有人担心不是专家的怎么办？人不可能一开始就是专家；对此，我觉得倒不必太过悲观，人需要大量的时间去积累才能成为一个领域的专家是过去的一种固定认知，随着AI助手的普及，人们并不需要在基础的知识存储和获取方面浪费太多时间，可以更专注于在AI助手的协助下，培养自己的领域知识鉴别能力、独立思考能力，快速完成从初级到高级人才的转变，或许一出学校就是领域专家不再是不可能的事。

## **小结**

NTP是一条更难走的路，chatgpt成功的因素：

•选择了NTP方向•获得了transformer-decoder模块的并行训练能力•算力和分布式训练技术的进步——参见工程篇•高质量训练数据 **在正确的方向上，才可能大力出奇迹** 下一步开始围绕应用工具、SFT等方向展开进一步阅读

## **OpenAI技术博客**

---

openai的blog提供了主要研究工作的概括性介绍，先浏览一遍，再精读paper（这部分也属于参考资料，单拎出来以和其他材料区分开来）

1.introducing ChatGPT：https://openai.com/blog/chatgpt

•要点•InstructGPT的sibling模型•paper：https://arxiv.org/abs/2203.02155•使用RLHF技术训练•数据集收集策略略有不同•奖励模型用了PPO策略•limitation（较多，选取几条，完整请参考原文）•ChatGPT 有时会写出看似合理但不正确或荒谬的答案。解决这个问题具有挑战性，因为：（1）在 RL 训练期间，目前没有真实来源； (2) 训练模型更加谨慎导致它拒绝可以正确回答的问题； (3) 监督训练会误导模型，因为理想的答案取决于模型知道什么，而不是人类演示者知道什么。

1.instructGPT

•要点•paper：https://arxiv.org/abs/2203.02155•GPT-3本身学术任务的表现优良•RLHF代表的alignment技术进一步解锁了模型的能力（为什么需要解锁？因为推理用到的算力和数据比训练少的多，模型大部分能力可能没有被激发出来）•平衡对齐和学术任务能力：使用RLHF微调的时候混合了一部分GPT-3的原始数据，并使用正常的对数似然最大化来训练，大致达到了对齐人类便好，并且保持利学术任务性能•Limitation•仍然可能在有指令或无指令的情况下生成有毒内容

1.RLHF

•要点•paper：https://arxiv.org/abs/1706.03741•构建安全 AI 系统的一个步骤是消除人类编写目标函数的需要，因为对复杂目标使用简单代理，或者将复杂目标弄错一点，可能会导致不良甚至危险的行为•通过人类反馈的RL比直接制定优化目标的RL要高效的多•RLHF的框架图如下所示：

![](https://pic1.zhimg.com/v2-037707e1eeaba69c06e1fc4ab5cf9e3c_b.jpg "null")

1.PPO

•要点•paper：https://arxiv.org/pdf/1707.06347.pdf•repo：https://github.com/openai/baselines•特点：•性能sota，实现和调优更简单•openai的默认RL算法•分析•传统策略梯度方法有效性不错，但对步长过于敏感•trpo：https://arxiv.org/abs/1502.05477•acer：https://arxiv.org/abs/1611.01224

## **参考**

---

1.语言模型历史：https://www.51cto.com/article/714730.html2.语言模型简介：https://zhuanlan.zhihu.com/p/434535483.神经网络语言模型：A Neural Probabilistic Language Model4.the annotated transformer：https://nlp.seas.harvard.edu/2018/04/03/attention.html5.pytorch版本transformer：http://nlp.seas.harvard.edu/annotated-transformer/6.算力单位pfs-day：https://zhuanlan.zhihu.com/p/1064064337.pfs-day官方：https://openai.com/research/ai-and-compute8.算力推导文章：https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e49.minibatch-sgd实现：https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/，用于理解activation 参数量；准确理解还需要LA基础10.微调11.显存推导：https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-models-memory12.https://huggingface.co/docs/transformers/perf_train_gpu_many13.很好的文档库：https://huggingface.co/docs/transformers/quicktour14.安倍架构材料：https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf15.ds-extreme model training：https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/16.3D并行训练：https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/#heading-3d-parallelism-scaling-to-trillion-parameter-models17.ds：https://www.microsoft.com/en-us/research/project/deepspeed/people/18.bloomchat-176B：https://huggingface.co/sambanovasystems/BLOOMChat-176B-v119.https://github.com/sambanova/bloomchat/tree/main/training20.The near future of AI is action-driven:https://jmcdonnell.substack.com/p/the-near-future-of-ai-is-action-driven21.chatgpt介绍：chatgpt网站22.full stack总结的LLM技术进展：https://mp.weixin.qq.com/s/weH_7K2g3sBMbtei1_dTng23.BPE介绍：https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee024.NLP中的迁移学习分类：https://www.ruder.io/state-of-transfer-learning-in-nlp/25.NLPCourse for u：https://lena-voita.github.io/nlp_course/transfer_learning.html26.GPT系列串讲：https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb227.gpt2原理：https://jalammar.github.io/illustrated-gpt2/28.decoder-only模型训练：https://ai.stackexchange.com/questions/40179/how-does-the-decoder-only-transformer-architecture-work29.gpt2模型说明：https://kknews.cc/code/zyoqq2q.html30.transformer的分类：https://cloud.tencent.com/developer/article/188582631.gpt-2代码走读：https://zhuanlan.zhihu.com/p/10823190432.需要补充tf1.x的知识33.gpt-2代码：https://github.com/openai/finetune-transformer-lm/blob/master/train.py34.https://yam.gift/2020/04/07/Paper/2020-04-07-GPT2/35.https://juejin.cn/post/684490401419940660036.teacher forcing技术：https://towardsdatascience.com/what-is-teacher-forcing-3da6217fed1c37.**复现gpt-2:http://fancyerii.github.io/2019/03/08/gpt2/38.RLHF：https://huggingface.co/blog/rlhf39.GPT-1、2详解：https://towardsdatascience.com/language-models-gpt-and-gpt-2-8bdb9867c50a40.state of gpt：https://m.youtube.com/watch?v=bZQun8Y4L2A&pp=ygUMc3RhdGUgb2YgZ3B041.chatgpt介绍：https://openai.com/blog/chatgpt42.openai模型命名列表：https://platform.openai.com/docs/model-index-for-researchers43.部署语言模型的一些经验教训：https://openai.com/research/language-model-safety-and-misuse

### **参考资料**

1: https://openai.com/blog/chatgpt

2: https://mp.weixin.qq.com/s?__biz=MzI1MTI2OTkxMA==&mid=2247484031&idx=1&sn=3ef4a0fc6dc4dbc41392ee0e33194387&chksm=e9f4c406de834d10d1545ee8ad918c24a0a37bf27e2c44cc3a0b77999b516c75f0bfe2eb191b&token=1408500559&lang=zh_CN#rd

3: https://openai.com/blog/chatgpt


