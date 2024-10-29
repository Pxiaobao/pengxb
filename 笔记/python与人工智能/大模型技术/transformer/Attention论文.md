Abstract:
	 提出了一个新的模型，完全基于注意力机制，完全摒弃递归和卷积。
递归的局限性：
	递归模型通常沿着输入和输出序列的符号位置来考虑计算。将位置与计算时间中的步骤对齐，它们生成一个隐藏状态序列ht，作为先前隐藏状态ht−1和位置t的输入的函数。这种固有的顺序性排除了训练示例中的并行化，这在较长的序列长度下变得至关重要，因为内存约束限制了示例之间的批处理。最近的工作通过因子分解技巧[21]和条件计算[32]显著提高了计算效率，同时也提高了后者的模型性能。然而，顺序计算的基本约束仍然存在。
注意力机制已成为各种任务中令人信服的序列建模和转导模型的组成部分，允许对依赖性进行建模，而不考虑它们在输入或输出序列中的距离。

 Background

	self-attention，有时称为intra-attention，是一种将单个序列的不同位置联系起来以计算序列的表示的注意机制。
==**multihead-attention ：类比如卷积神经网络来看，每次卷积只能看到很小的一个窗口，比如3*3,如果要看到两个距离比较远的像素点的话，需要多次卷积，才能聚合到一起；**==
==**但是对于transformer的Attention来说，只需要一次既可以看到所有的像素；当然了卷积每次可以有多个通道，每个通道提取不同的特征，类似的transformer中的multi-head Attention也起到了类似的效果。**==


Model Architecture:

	Most competitive neural sequence transduction models have an encoder-decoder structure.Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive , consuming the previously generated symbols as additional input when generating the next.
	
	大多数竞争性神经序列转导模型都具有编码器-解码器结构。这里，编码器将符号表示的输入序列（x1，…，xn）映射到连续表示的序列z＝（z1，…，zn）。给定z（z为向量），解码器然后一次一个元素地生成符号的输出序列（y1，…，ym）。在每一步，模型都是自回归的，在生成下一步时，将先前生成的符号作为额外输入。

	![[Pasted image 20240703160846.png]]The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,respectively.
	Transformer遵循enconder-deconder这一总体架构，使用堆叠的self-attention和point-wise、完全连接的编码器和解码器层，分别如图1的左半部分和右半部分所示。
	
	We also modify the self-attention
	sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
	masking, combined with fact that the output embeddings are offset by one position, ensures that the
	predictions for position i can depend only on the known outputs at positions less than i.
	我们还修改了解码器堆栈中的自注意子层，以防止位置关注后续位置。这种掩蔽，再加上输出嵌入偏移一个位置的事实，确保了对位置i的预测只能取决于小于i的位置处的已知输出。


3.1 Encoder and Decoder Stacks
编码器：编码器由N=6个相同层的堆栈组成。每一层有两个子层。第一种是多头自我注意机制，第二种是简单的、位置上完全连接的前馈网络。我们在两个子层的每一个周围使用残差连接[11]，然后进行层归一化[1]。也就是说，每个子层的输出是LayerNorm（x+子层（x）），其中子层（x）是子层本身实现的函数。为了促进这些残余连接，模型中的所有子层以及嵌入层都会产生维度dmodel=512的输出。

==LayerNorm：与batchNorm对比，以二维输入为例，每一列代表一个feature，做归一化的时候batchNorm是取每一列做标准化，LayerNorm是取每一行做标准化；==
作用是：算均值和方差的时候，
![[Pasted image 20240731144134.png]]

解码器：解码器==也由N=6个相同层的堆栈组成==。除了每个编码器层中的两个子层外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围使用残差连接，然后进行层归一化。我们还修改了解码器堆栈中的自关注子层，以防止位置关注后续位置。这种掩蔽，再加上输出嵌入偏移了一个位置的事实，确保了位置i的预测只能依赖于小于i的位置处的已知输出。


 3.2 Attention：
 ![[Pasted image 20240709164039.png]]
	 An attention function can be described as mapping a query and a set of key-value pairs to an output,where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
	 注意力函数可以被描述为将查询和一组键值对映射到输出，其中q、k、v和输出都是向量。输出为v的加权和，其中分配给k的权重是通过q与对应关键字的相似度来计算的。

		3.2.1 scaled Dot-product attention
		余弦相似度表示，内积越大 相似度越大，内积为0，正交 完全不同。
![[Pasted image 20240731145123.png]]

		如何做mask,保证如何在t时刻，只能看到t时刻之前的东西？
		3.2.2 multi-head attention
![[Pasted image 20240731150121.png]]
		3.2.3 
3.3 Position-wise Feed-Forword NetWorks
mlp：transformer与rnn的区别
![[Pasted image 20240801153718.png]]
	
	3.4 Embeddings and Softmax
	
	
	
	3.5 Positional Encoding
	通过sin\cos编码将每个向量的位置信息加入进去。


4、why self-attention
![[Pasted image 20240801154927.png]]
为什么要使用自注意力，相比了循环和卷积、

7：conclusion
	第一次提出multi-headed self-attention
	