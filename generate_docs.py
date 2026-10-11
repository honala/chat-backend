"""
知识库文档生成器 — 生成50篇模拟专业文档用于RAG系统测试
主题：人工智能通识课程知识库
"""
import json
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("HONALA_KNOWLEDGE_DOCS_DIR", os.path.join(os.path.dirname(__file__), "knowledge_docs"))).resolve()

DOCUMENTS = [
    # ── 01-10: AI 基础概念 ──
    {
        "title": "人工智能概述",
        "content": """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够模拟人类智能行为的系统。
AI的核心目标包括：推理、知识表示、规划、学习、自然语言处理、感知以及物体操作等能力。

人工智能的发展经历了多个阶段：
1. 萌芽期（1943-1956）：图灵测试的提出，达特茅斯会议的召开标志着AI的正式诞生。
2. 黄金期（1956-1974）：早期AI程序如Logic Theorist、ELIZA等出现。
3. 第一次寒冬（1974-1980）：由于计算能力限制和期望过高，AI研究资金大幅削减。
4. 专家系统时代（1980-1987）：基于规则的专家系统在商业领域取得成功。
5. 第二次寒冬（1987-1993）：专家系统的局限性暴露，AI再次陷入低谷。
6. 现代AI（1993至今）：机器学习、深度学习技术的突破推动了AI的飞速发展。

当前AI的主要研究方向包括：
- 机器学习：让计算机通过数据学习模式
- 深度学习：利用多层神经网络处理复杂任务
- 自然语言处理：使计算机理解和生成人类语言
- 计算机视觉：让计算机理解图像和视频内容
- 强化学习：通过与环境交互学习最优策略
- 知识图谱：构建结构化的知识表示系统"""
    },
    {
        "title": "机器学习基础",
        "content": """机器学习（Machine Learning）是人工智能的核心子领域，研究如何让计算机系统利用数据来改善性能。
机器学习的本质是从数据中自动发现模式和规律，并利用这些规律对未来数据进行预测或决策。

机器学习的三大范式：
1. 监督学习（Supervised Learning）：使用带标签的训练数据来学习输入到输出的映射关系。
   常见算法：线性回归、逻辑回归、决策树、随机森林、支持向量机（SVM）、K近邻（KNN）、朴素贝叶斯。
   典型应用：垃圾邮件分类、房价预测、图像识别、疾病诊断。

2. 无监督学习（Unsupervised Learning）：从无标签数据中发现隐藏的结构和模式。
   常见算法：K-Means聚类、层次聚类、DBSCAN、主成分分析（PCA）、t-SNE、自编码器（Autoencoder）。
   典型应用：客户分群、异常检测、降维可视化、推荐系统中的协同过滤。

3. 强化学习（Reinforcement Learning）：智能体通过与环境交互，根据奖励信号学习最优行为策略。
   核心概念：状态（State）、动作（Action）、奖励（Reward）、策略（Policy）、价值函数（Value Function）。
   经典算法：Q-Learning、DQN（Deep Q-Network）、策略梯度（Policy Gradient）、PPO（Proximal Policy Optimization）。
   里程碑应用：AlphaGo击败围棋世界冠军、OpenAI Five在Dota 2中表现出色。

机器学习的核心挑战：
- 过拟合（Overfitting）：模型在训练数据上表现优异，但在新数据上泛化能力差
- 欠拟合（Underfitting）：模型过于简单，无法捕捉数据的内在规律
- 数据质量：脏数据、缺失值、不平衡数据严重影响模型性能
- 特征工程：如何选择和构造有效的特征是一个关键问题
- 模型可解释性：复杂模型如同黑盒，难以解释其决策过程"""
    },
    {
        "title": "深度学习原理",
        "content": """深度学习（Deep Learning）是机器学习的一个子集，基于人工神经网络，特别是具有多个隐藏层的深层神经网络。
深度学习的"深度"指的是网络中层的数量，深层架构使得模型能够学习数据的层次化特征表示。

人工神经网络的基础组成：
1. 神经元（Neuron）：受生物神经元启发的计算单元，接收多个输入并产生一个输出。
   每个神经元执行：加权求和 + 偏置 + 激活函数。

2. 激活函数（Activation Function）：引入非线性变换，使网络能够拟合复杂函数。
   - Sigmoid：将输出压缩到(0,1)，适用于二分类输出层
   - Tanh：将输出压缩到(-1,1)，中心化数据，常用于隐藏层
   - ReLU：f(x)=max(0,x)，计算简单，缓解梯度消失问题
   - Leaky ReLU：解决ReLU的"神经元死亡"问题
   - Softmax：将多个输出转换为概率分布，用于多分类

3. 损失函数（Loss Function）：衡量模型预测与真实值之间的差距。
   - 均方误差（MSE）：回归任务
   - 交叉熵（Cross-Entropy）：分类任务
   - Hinge Loss：支持向量机

4. 优化算法（Optimizer）：通过梯度下降更新网络权重以最小化损失。
   - SGD（随机梯度下降）
   - Momentum（动量法）
   - Adam（自适应矩估计）
   - RMSprop

5. 反向传播（Backpropagation）：计算损失函数对每个参数的梯度，从输出层向输入层逆向传播误差。

主流深度学习架构：
- 卷积神经网络（CNN）：擅长处理网格化数据（如图像），通过卷积核提取局部特征
- 循环神经网络（RNN/LSTM/GRU）：处理序列数据，具有记忆能力
- Transformer：基于自注意力机制，已成为NLP和CV领域的主流架构
- 生成对抗网络（GAN）：生成器与判别器对抗训练，用于图像生成等任务
- 图神经网络（GNN）：处理图结构数据，适用于社交网络分析和分子结构预测"""
    },
    {
        "title": "自然语言处理入门",
        "content": """自然语言处理（Natural Language Processing，NLP）是AI与语言学的交叉领域，致力于使计算机能够理解、解释和生成人类语言。

NLP的核心任务层次：
1. 词法分析：分词、词性标注、命名实体识别（NER）
2. 句法分析：依存句法分析、成分句法分析
3. 语义分析：词义消歧、语义角色标注、情感分析
4. 语用分析：指代消解、对话意图识别
5. 篇章分析：文本摘要、主题建模、信息抽取

文本预处理的关键步骤：
- 分词（Tokenization）：将连续文本切分为词或子词单元
- 去除停用词：移除"的""是""在"等高频低信息量词汇
- 词干提取/词形还原：将词语还原为基本形式
- 文本向量化：TF-IDF、Word2Vec、GloVe、BERT Embedding

传统NLP方法与现代方法的对比：
传统方法依赖手工特征和规则，如基于TF-IDF的文本分类、基于HMM的序列标注。
现代方法以预训练语言模型为核心，如BERT、GPT系列，通过大规模无监督预训练+下游微调的两阶段范式，在各种NLP任务上取得突破性成果。

中文NLP的特殊挑战：
- 中文没有天然的空格分词边界，需要专门的分词工具（如jieba）
- 汉字的一字多义和多字近义现象增加了语义理解的难度
- 中文存在大量成语、俗语和网络新词"""
    },
    {
        "title": "计算机视觉概述",
        "content": """计算机视觉（Computer Vision）是AI的重要分支，旨在使计算机能够从图像和视频中获取高层次的语义理解。

计算机视觉的核心任务：
1. 图像分类（Image Classification）：判断图像所属类别
2. 目标检测（Object Detection）：定位并识别图像中的多个物体
3. 语义分割（Semantic Segmentation）：对图像中的每个像素进行分类
4. 实例分割（Instance Segmentation）：区分同一类别中的不同个体
5. 图像生成（Image Generation）：根据描述或噪声生成新图像
6. 姿态估计（Pose Estimation）：识别人体或物体的关键点和姿态

卷积神经网络（CNN）的核心组件：
- 卷积层（Convolutional Layer）：通过可学习的卷积核提取局部特征
- 池化层（Pooling Layer）：降低特征图的空间维度，增加感受野
- 全连接层（Fully Connected Layer）：将提取的特征映射到最终输出

经典视觉模型演进：
2012年：AlexNet在ImageNet竞赛中大幅领先，开启了深度学习视觉时代
2014年：VGGNet使用小卷积核堆叠，证明了网络深度的重要性
2015年：ResNet引入残差连接，解决了深层网络的退化问题，可达152层
2017年：DenseNet提出密集连接，进一步改善梯度流动
2020年：Vision Transformer（ViT）将Transformer架构引入视觉领域
2024年：多模态大模型统一理解和生成，如GPT-4V、CLIP、DALL-E等

数据增强技术：
- 翻转、旋转、缩放、裁剪等几何变换
- 颜色抖动、亮度调整等色彩变换
- MixUp、CutMix等高级混合增强方法"""
    },
    {
        "title": "大语言模型技术概述",
        "content": """大语言模型（Large Language Model，LLM）是基于Transformer架构的大规模预训练语言模型，
通过在海量文本数据上进行自监督学习，获得强大的语言理解和生成能力。

LLM的核心技术特征：
1. 规模定律（Scaling Law）：模型性能与参数量、数据量、计算量呈幂律关系
2. 涌现能力（Emergent Abilities）：当模型规模超过某个阈值时，突然展现出训练时未明确教导的能力
3. 上下文学习（In-Context Learning）：无需参数更新，仅通过提示中的少量示例即可适应新任务

主流大语言模型：
- GPT系列（OpenAI）：GPT-3（175B参数）、GPT-4（多模态，参数未公开）
- Claude系列（Anthropic）：强调安全性和有用性的平衡
- Gemini系列（Google）：原生多模态架构
- LLaMA系列（Meta）：开源大模型，从7B到405B参数
- DeepSeek系列（深度求索）：高效推理，DeepSeek-V3采用MoE架构
- Qwen系列（阿里）：通义千问，支持中英文等多语言
- GLM系列（智谱）：ChatGLM在中文任务上表现出色
- Mistral系列（Mistral AI）：高效开源模型

大语言模型的训练流程：
1. 预训练（Pre-training）：在海量文本上进行下一个词预测，学习通用语言知识
2. 有监督微调（SFT）：使用高质量的指令-回答对进行微调，提升指令跟随能力
3. 人类反馈强化学习（RLHF）：通过人类偏好反馈训练奖励模型，用PPO优化生成质量
4. 直接偏好优化（DPO）：简化RLHF流程，直接优化偏好对

LLM的关键性能指标：
- 困惑度（Perplexity）：评估语言模型质量
- BLEU/ROUGE：评估生成文本与参考文本的相似度
- 人类评估：真实性和有用性的主观判断
- 基准测试：MMLU（多任务语言理解）、HumanEval（代码生成）等"""
    },
    {
        "title": "Transformer架构详解",
        "content": """Transformer是由Google在2017年论文"Attention Is All You Need"中提出的革命性神经网络架构，
彻底改变了自然语言处理领域，并迅速扩展到计算机视觉、语音识别等多个AI子领域。

Transformer的核心创新——自注意力机制（Self-Attention）：
- 每个位置的表示通过关注（attending to）序列中的所有位置来计算
- 公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Q（Query）：当前词想要查询什么信息
- K（Key）：每个词携带的标签信息
- V（Value）：每个词实际的内容信息
- 缩放因子√d_k防止点积过大导致softmax梯度消失

多头注意力（Multi-Head Attention）：
- 并行运行多个注意力头，每个头关注不同的表示子空间
- 类似于CNN中的多个卷积核，捕获多种特征
- 最终将各头输出拼接并通过线性变换融合

位置编码（Positional Encoding）：
由于Transformer没有循环或卷积结构，需要显式编码位置信息。
- 正弦位置编码：使用固定频率的正弦/余弦函数
- 可学习位置编码：将位置作为可训练参数
- 旋转位置编码（RoPE）：通过旋转矩阵编码相对位置关系

Transformer的变体架构：
1. 仅编码器（Encoder-only）：BERT系列，适用于理解任务
2. 仅解码器（Decoder-only）：GPT系列，适用于生成任务
3. 编码器-解码器（Encoder-Decoder）：T5、BART，适用于翻译等任务

Transformer的训练技巧：
- 层归一化（Layer Normalization）：稳定训练，加速收敛
- 残差连接（Residual Connection）：缓解梯度消失，支持深层堆叠
- Dropout：防止过拟合
- 学习率预热（Warmup）：训练初期逐步增加学习率
- 梯度裁剪（Gradient Clipping）：防止梯度爆炸"""
    },
    {
        "title": "注意力机制深入",
        "content": """注意力机制（Attention Mechanism）是现代深度学习的核心组件，其灵感来源于人类视觉注意力——
我们不会平等地处理视野中的所有信息，而是聚焦于关键区域。

注意力机制的数学本质：
计算一组值（Values）的加权和，权重由查询（Query）和键（Key）的相似度决定。
可以理解为一种"软寻址"机制：根据查询内容，从记忆中检索相关信息。

注意力机制的主要类型：
1. 加性注意力（Additive Attention / Bahdanau Attention）：
   使用前馈网络计算注意力分数，最早应用于机器翻译。
   score(q,k) = v^T * tanh(W_q * q + W_k * k)

2. 点积注意力（Dot-Product Attention）：
   直接使用向量点积，计算更快，是Transformer的基础。
   score(q,k) = q^T * k

3. 缩放点积注意力（Scaled Dot-Product Attention）：
   为点积注意力添加缩放因子√d_k，防止维度增大导致softmax进入饱和区。

4. 自注意力（Self-Attention）：
   查询、键、值都来自同一个序列，用于建模序列内部依赖关系。
   这是Transformer成功的关键。

5. 交叉注意力（Cross-Attention）：
   查询来自一个序列，键和值来自另一个序列。
   用于编码器-解码器架构中的解码器，实现对编码器输出的关注。

6. 稀疏注意力（Sparse Attention）：
   限制每个位置关注的上下文范围，降低计算复杂度从O(n^2)到O(n)。
   变体：滑动窗口注意力、扩张注意力、全局+局部注意力。

注意力机制的可视化与分析：
- 注意力热力图展示了模型在生成每个输出词时关注的输入部分
- 不同注意力头学习到不同的语言现象：有的关注相邻词，有的关注远距离依赖
- BERT的注意力分析揭示了模型隐式学习了句法结构"""
    },
    {
        "title": "知识图谱技术",
        "content": """知识图谱（Knowledge Graph）是一种结构化的语义知识库，以图的形式描述现实世界中的实体、概念及其关系。
知识图谱将信息组织为（头实体，关系，尾实体）的三元组形式，如：（北京，是首都，中国）。

知识图谱的构建流程：
1. 知识获取：从结构化、半结构化和非结构化数据中提取知识
2. 知识融合：整合多个来源的知识，解决实体冲突和冗余
3. 知识表示：将实体和关系映射到低维向量空间（知识图谱嵌入）
4. 知识推理：基于已有知识推导新的知识
5. 知识更新：随着新知识的产生不断更新图谱

知识图谱嵌入方法：
- TransE：将关系建模为头实体到尾实体的平移向量，h + r ≈ t
- TransR：在不同关系空间中分别建模实体和关系
- RotatE：将关系建模为复数空间中的旋转
- ComplEx：在复数空间中建模非对称关系

知识图谱的典型应用：
- 搜索引擎增强：Google知识图谱为搜索结果提供结构化信息卡片
- 智能问答：基于知识图谱推理回答复杂问题
- 推荐系统：利用知识图谱中的实体关联增强推荐效果
- 药物发现：利用生物医学知识图谱加速新药研发
- 金融风控：构建企业关联图谱，识别潜在风险"""
    },
    {
        "title": "AI伦理与安全",
        "content": """随着人工智能技术的快速发展，AI伦理与安全问题日益成为学术界和产业界的关注焦点。
负责任的AI发展需要在技术创新与伦理约束之间找到平衡。

AI伦理的核心原则：
1. 公平性（Fairness）：AI系统不应因种族、性别、年龄等因素产生歧视性结果
2. 透明度（Transparency）：AI决策过程应当可解释，用户应知情理解
3. 隐私保护（Privacy）：在数据收集和使用过程中保护个人信息
4. 安全性（Safety）：确保AI系统在各种条件下安全可靠运行
5. 问责制（Accountability）：明确AI系统造成损害时的责任归属

AI对齐（AI Alignment）问题：
- 价值对齐：确保AI系统的目标与人类价值观一致
- 意图对齐：AI系统理解并执行人类的真实意图，而非字面指令
- 奖励建模：设计能够正确引导AI行为的奖励函数

偏见来源与缓解：
- 数据偏见：训练数据中的历史偏见被模型学习放大
- 算法偏见：模型架构和优化目标引入的偏见
- 部署偏见：AI系统在不同场景下可能产生不同效果
缓解方法：数据平衡采样、公平性约束训练、对抗去偏、后处理校准

AI安全技术：
- 对抗鲁棒性：抵御对抗样本攻击
- 差分隐私：在数据分析中保护个体隐私
- 联邦学习：在分散数据上训练模型，不共享原始数据
- 红队测试（Red Teaming）：主动寻找AI系统的漏洞和风险"""
    },
    # ── 11-20: RAG技术专题 ──
    {
        "title": "RAG技术原理",
        "content": """检索增强生成（Retrieval-Augmented Generation，RAG）是一种将信息检索与文本生成相结合的AI架构，
由Facebook AI Research（现Meta AI）在2020年提出。RAG通过引入外部知识库，有效缓解了大语言模型的幻觉问题。

RAG的核心工作流程包含三个阶段：

一、索引阶段（Indexing）
1. 文档加载：支持多种格式（PDF、Word、HTML、TXT等）的文档解析，提取纯文本内容
2. 文本分块（Chunking）：将长文档切分为适当大小的文本块
   - Chunk Size：每个文本块的字符数或token数，常见设置512-1024
   - Chunk Overlap：相邻块之间的重叠字符数，通常为10-20%的chunk_size
   - 分块策略：固定大小分块、按段落/句子边界分块、语义分块
3. 向量化（Embedding）：使用Embedding模型将每个文本块转换为固定维度的稠密向量
   - 常用Embedding模型：OpenAI text-embedding-ada-002、BGE系列、Nomic Embed Text、Jina Embeddings
   - 向量维度的选择影响存储成本和检索精度
4. 向量存储：将生成的向量及元数据存入向量数据库

二、检索阶段（Retrieval）
1. 查询向量化：将用户问题通过同一Embedding模型转换为查询向量
2. 相似度计算：在向量数据库中执行近似最近邻（ANN）搜索
   - 余弦相似度：最常用的语义相似度度量
   - 欧氏距离：适用于归一化向量空间
   - 点积：计算效率高，常用于大规模检索
3. Top-K检索：返回相似度最高的K个文档片段"""
    },
    {
        "title": "RAG系统架构设计",
        "content": """一个生产级的RAG系统通常包含以下核心模块，各模块协同工作以提供高质量的检索增强生成服务。

一、数据层
1. 文档存储：原始文档的文件系统或对象存储（MinIO、AWS S3）
2. 文档元数据：文档名称、上传时间、分类标签、权限信息等
3. 向量数据库：存储文档块的Embedding向量及索引
4. 关系数据库：存储用户信息、对话历史、反馈数据

二、处理层
1. 文档解析器：多格式文档的文本提取引擎
2. 文本预处理：清洗、分词、去噪等操作
3. 智能分块器：根据文档结构和语义自适应调整分块策略
4. Embedding服务：统一的向量化接口，支持模型切换
5. 检索器：语义检索 + 关键词检索的混合检索引擎

三、服务层
1. API网关：统一的API入口，负责认证、限流、路由
2. 查询优化器：查询重写、查询扩展、查询分解
3. 重排序器（Reranker）：对初步检索结果进行精细排序
4. 上下文组装器：将检索结果与对话历史组装为Prompt
5. LLM调用器：负责与大语言模型API的通信

四、应用层
1. Web交互界面：文档上传、知识库管理、问答交互
2. 反馈收集：用户点赞/点踩、答案评分
3. 监控告警：检索质量监控、系统性能监控
4. 日志审计：记录所有查询和响应，用于分析和优化

关键设计考量：
- 扩展性：向量数据库应支持水平扩展以应对文档增长
- 实时性：新文档索引后应尽快可检索
- 可靠性：提供降级策略，如检索失败时直接使用LLM通用知识回答
- 安全性：文档权限控制、API密钥管理、敏感信息过滤"""
    },
    {
        "title": "Embedding模型对比分析",
        "content": """Embedding模型是RAG系统的核心组件，其质量直接影响检索效果。以下是主流Embedding模型的详细对比。

一、开源Embedding模型

1. BGE系列（BAAI）
   - bge-large-zh-v1.5：中文Embedding的标杆，1024维，MTEB中文榜Top
   - bge-m3：多语言版本，支持100+语言，支持稠密+稀疏混合检索
   - bge-en-icl：专为上下文学习优化的英文模型
   优点：中文效果最佳，支持指令微调
   缺点：模型较大（~1.3GB），推理速度较慢

2. Nomic Embed Text
   - nomic-embed-text-v1.5：768维，支持长文本（8192 tokens）
   - 可与Ollama本地部署，适合离线场景
   优点：开源可商用，支持长文本
   缺点：中文效果略逊于专门的中文模型

3. Jina Embeddings
   - jina-embeddings-v3：8192 tokens上下文，570M参数
   - 支持任务特定LoRA适配器
   优点：长上下文，任务适配灵活
   缺点：模型较大

4. Multilingual-E5
   - multilingual-e5-large：1024维，支持多语言
   - 训练方式：弱监督对比学习
   优点：多语言统一表示空间
   缺点：需要前缀"query:"和"passage:"指定模式

二、商业Embedding服务

1. OpenAI text-embedding-3-large：3072维（可缩减），支持多语言
2. Cohere Embed v3：1024维，专为RAG优化
3. Voyage AI voyage-2：1024维，长文档检索表现优异

三、Embedding模型选型建议
- 中文场景：首选bge-large-zh-v1.5或bge-m3
- 本地离线：Nomic Embed Text + Ollama
- 长文档检索：Jina Embeddings v3或Nomic
- 成本敏感：bge-small-zh-v1.5（512维）"""
    },
    {
        "title": "向量数据库选型指南",
        "content": """向量数据库是RAG系统的存储和检索核心，选择合适的向量数据库对系统性能和成本有重大影响。

一、主流向量数据库对比

1. Chroma
   - 类型：开源嵌入式向量数据库
   - 存储引擎：支持DuckDB（本地）和ClickHouse（生产）
   - 索引：HNSW，支持余弦/欧氏/IP距离
   - 优势：Python原生，API简洁，部署零配置，适合快速原型开发
   - 局限：单机部署，大规模场景性能瓶颈，社区版功能有限
   - 适用场景：中小规模RAG应用、原型验证、学习研究

2. FAISS（Facebook AI Similarity Search）
   - 类型：向量检索库（非完整数据库）
   - 索引：IVF、HNSW、PQ量化等多种索引
   - 优势：GPU加速、检索速度极快、Meta维护活跃
   - 局限：仅提供检索功能，需自行管理元数据和持久化
   - 适用场景：对检索速度有极致要求的场景

3. Milvus
   - 类型：云原生分布式向量数据库
   - 架构：计算存储分离，支持水平扩展
   - 索引：支持11种索引类型
   - 优势：高性能、高可用、完善的监控生态
   - 局限：部署复杂、资源消耗大
   - 适用场景：大规模生产级RAG应用

4. Weaviate
   - 类型：开源向量数据库 + 知识图谱
   - 特点：内建向量化和混合搜索，GraphQL接口
   - 优势：功能全面，支持多模态
   - 适用场景：需要知识图谱关联的复杂RAG系统

5. Qdrant
   - 类型：高性能向量数据库
   - 语言：Rust编写，性能优异
   - 优势：高级过滤、量化压缩、水平扩展
   - 适用场景：对性能要求高的生产环境

二、选型决策树
1. 数据量 < 10万条 + 单机 → Chroma
2. 数据量 10万-100万 + 性能优先 → Qdrant
3. 数据量 > 100万 + 需要扩展 → Milvus
4. 需要GPU加速 → FAISS
5. 需要知识图谱 → Weaviate"""
    },
    {
        "title": "RAG检索优化策略",
        "content": """RAG系统的检索质量决定了最终答案的准确性和完整性。以下是经过实践验证的检索优化策略。

一、混合检索（Hybrid Search）
单一检索方式的局限性：
- 纯语义检索（向量相似度）可能遗漏精确关键词匹配
- 纯关键词检索（如BM25）无法理解语义相近但表达不同的内容

混合检索结合两者优势：
1. 分别执行语义检索和关键词检索
2. 使用融合算法合并两路结果
   - 倒数排序融合（RRF）：score(d) = Σ 1/(k + rank_i(d))
   - 线性加权融合：score(d) = α * semantic_score + (1-α) * keyword_score

二、重排序（Reranking）
初步检索返回Top-N结果（如50条），再用更精细的模型重新排序：
- Cross-Encoder Reranker：将(query, document)对输入模型，输出相关性分数
- BGE-Reranker：bge-reranker-v2-m3，专为RAG优化
- Cohere Rerank：商业服务，效果优异
- LLM-based Reranking：使用LLM直接评估文档相关性

三、查询优化
1. 查询重写（Query Rewriting）：将口语化问题改写为更精确的检索查询
2. 查询分解（Query Decomposition）：将复杂问题拆分为多个子问题分别检索
3. 假设文档嵌入（HyDE）：先用LLM生成假设答案，用假设答案的向量替代原问题进行检索
4. 多轮查询（Multi-Query）：生成多个不同表述的查询，提升覆盖度

四、分块策略优化
1. 小→大检索（Small-to-Big）：用小块检索，返回大块上下文
2. 句子窗口检索：检索单个句子，返回周围窗口内的上下文
3. 父子文档（Parent-Child）：检索子块，返回父文档完整上下文

五、MMR（最大边际相关性）
在保证语义相关性的同时，最大化结果之间的多样性：
MMR = argmax[λ * sim(D_i, Q) - (1-λ) * max(sim(D_i, D_j))]
其中λ控制相关性和多样性的权衡"""
    },
    {
        "title": "Prompt工程与上下文组装",
        "content": """在RAG系统中，检索结果的上下文组装和Prompt构建是连接检索与生成的关键桥梁。
一个好的Prompt能将检索到的分散信息组织为LLM易于理解和利用的格式。

一、Prompt模板设计原则
1. 角色设定：明确定义AI助手的角色和职责
2. 任务指令：清晰地描述期望的输出格式和内容要求
3. 知识约束：指明哪些信息可以用，哪些不能用
4. 引用规范：要求模型标注信息来源，增强可信度

二、上下文组装策略
1. 简单拼接法：将所有检索结果按相似度降序直接拼接
   - 优点：实现简单
   - 缺点：可能导致上下文过长，重要信息被截断

2. 结构化组装：按照文档、段落、重要度的层次结构组织
   - 每个文档块带编号，方便引用
   - 在处理冲突信息时明确标注来源

3. 选择性组装：根据相关性阈值过滤低质量结果
   - 设置相似度最低阈值（如0.6），过滤弱相关文档
   - 控制上下文总长度不超过模型窗口的60-70%

4. 分级组装：
   - 第一级：高度相关的结果全文放入
   - 第二级：中等相关的结果放入摘要
   - 第三级：弱相关的结果仅提供标题和来源

三、典型RAG Prompt模板

系统提示词示例：
\"你是一个基于知识库的智能问答助手。请根据以下知识库文档内容回答用户问题。

回答规则：
1. 优先使用知识库中的信息，引用时使用 [文档编号] 标注
2. 如果知识库信息不足以回答，请明确说明并给出你的建议
3. 如果多个来源有冲突信息，请指出并分别说明
4. 回答结尾列出\"## 参考来源\"小节

知识库内容：
---（此处插入检索结果）---\""""
    },
    {
        "title": "RAG与LLM幻觉的关系",
        "content": """大语言模型的幻觉（Hallucination）问题是指模型生成看似合理但与事实不符或完全虚构的内容。
RAG被广泛认为是缓解幻觉问题的有效方法，但其效果也受到多种因素的影响。

一、LLM幻觉的主要类型
1. 事实性幻觉：生成的内容与客观事实不符
   - 例如：声称某人在世实际上已去世，虚构不存在的论文
2. 忠实性幻觉：生成的内容偏离了用户指令或上下文
   - 例如：要求总结A文档，却编造了文档中不存在的信息
3. 逻辑性幻觉：推理过程存在逻辑漏洞
   - 例如：因果倒置、循环论证

二、RAG如何缓解幻觉
1. 提供事实锚定：检索到的真实文档为生成提供事实依据
2. 约束生成空间：模型被引导在检索结果范围内生成，减少自由发挥
3. 来源可追溯：用户可以验证答案的信息来源，形成事实核查闭环
4. 知识实时更新：无需重新训练模型，通过更新知识库即可获得最新信息

三、RAG的局限性
1. 检索质量依赖：如果检索不到相关文档，RAG的增益为零
2. 噪声引入：检索到不相关或低质量文档反而可能误导模型
3. 忠实性问题：模型可能忽视检索结果，仍依赖自身预训练知识
4. 上下文长度限制：大量检索结果可能超出模型的上下文窗口
5. 矛盾处理不足：当多个来源信息矛盾时，模型可能选择性地忽略

四、增强RAG防幻觉的策略
- 知识验证：用Cross-Encoder验证检索结果与问题的相关性
- 引用强制：在Prompt中明确要求每句话标注来源
- 置信度标注：要求模型对不确定的信息标注置信度
- 答案验证：使用第二个模型验证生成的答案是否忠实于检索结果
- 拒答机制：当检索结果置信度低时，明确告知用户无法回答"""
    },
    {
        "title": "多模态RAG技术前沿",
        "content": """多模态RAG（Multimodal RAG）将传统文本RAG的能力扩展到图像、音频、视频、表格等多种数据格式，
使得系统能够理解和检索跨模态的信息。

一、多模态RAG的核心挑战
1. 多模态Embedding对齐：将不同模态的数据映射到统一的向量空间
2. 跨模态检索：文本查询检索图像，或图像查询检索文本
3. 多模态上下文组装：如何在Prompt中有效组织文本+图像+表格等混合内容

二、关键技术组件
1. 多模态Embedding模型：
   - CLIP系列（OpenAI）：图文对比学习，支持文搜图、图搜文
   - ImageBind（Meta）：绑定六种模态（图像、文本、音频、深度、热感、IMU）
   - BridgeTower：融合视觉和文本的Transformer架构

2. 多模态文档解析：
   - pdf2image + OCR：将扫描PDF解析为文本
   - LayoutLM系列：理解文档的版面布局（标题、段落、表格、图片位置）
   - Unstructured.io：开源的多格式文档解析工具包

3. 多模态向量存储：
   - 为不同模态建立独立的Collection或索引
   - 使用多向量表示：每个文档块同时存储文本向量和图像向量

三、典型应用场景
1. 智能客服：用户上传产品照片+文字描述，系统自动检索相关文档
2. 医学诊断：结合医学影像和病历文本进行综合检索
3. 法律文书：解析含表格和图表的合同、判决书
4. 教育辅助：根据教材中的图表和公式进行智能问答
5. 电商搜索：图片+文字的组合搜索，返回相关商品

四、前沿发展方向
- 视频RAG：对长视频进行时序分块和索引
- 3D场景理解：检索和生成三维场景描述
- 实时多模态交互：语音+视觉的实时RAG系统"""
    },
    {
        "title": "RAG评估方法与指标",
        "content": """科学全面的评估是RAG系统优化的基础。RAG评估需要同时关注检索质量、生成质量和端到端效果三个维度。

一、检索质量评估
1. 命中率（Hit Rate）：Top-K检索结果中包含正确答案的比例
2. 平均倒数排名（MRR）：第一个相关文档排名的倒数平均值
   MRR = (1/|Q|) * Σ 1/rank_i
3. NDCG（Normalized Discounted Cumulative Gain）：
   考虑排序位置的归一化折损累计增益
4. 召回率@K（Recall@K）：Top-K结果中相关文档占所有相关文档的比例
5. 精确率@K（Precision@K）：Top-K结果中相关文档的比例

二、生成质量评估
1. 答案忠实度（Faithfulness）：
   - 评估生成的答案是否忠实于检索到的上下文
   - 使用自然语言推理（NLI）模型判断是否可以从上下文推断出答案
2. 答案相关性（Answer Relevance）：
   - 评估答案是否直接回应了用户的问题
   - 可通过生成反向问题并与原问题比较来衡量
3. 上下文相关性（Context Relevance）：
   - 评估检索到的上下文是否与问题相关
   - 可通过计算上下文与问题/答案的语义相似度来评估

三、端到端评估框架
1. RAGAS（RAG Assessment）：
   - 无需人工标注的自动化评估框架
   - 核心指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall
2. ARES（Automated RAG Evaluation System）：
   - 使用合成数据训练的评估模型
3. TruLens：
   - 提供RAG三元组评估：Answer Relevance、Context Relevance、Groundedness

四、人工评估维度
1. 正确性：答案是否事实正确
2. 完整性：答案是否回答了问题的所有方面
3. 有用性：答案是否对用户有帮助
4. 可读性：答案的语言表达是否清晰流畅"""
    },
    {
        "title": "RAG vs 微调对比分析",
        "content": """RAG（检索增强生成）和模型微调（Fine-tuning）是两种主要的LLM定制化方法，
它们适用于不同的场景，也可以互补使用。

一、技术本质的差异
RAG：
- 不修改模型参数，通过外部知识库增强模型能力
- 在推理时动态检索相关信息
- 知识是"外挂"的，可以随时更新

微调：
- 修改模型参数，让模型内化特定领域的知识
- 在训练时学习，推理时直接使用
- 知识是"内化"的，更新需要重新训练

二、适用场景对比

RAG更适用的场景：
1. 知识频繁更新：法律法规、新闻资讯、产品文档
2. 需要来源追溯：需要明确标注信息来源
3. 多知识库隔离：不同用户/场景使用不同的知识库
4. 快速原型验证：无需训练，快速上线

微调更适用的场景：
1. 风格/格式统一：让模型以特定格式和风格输出
2. 复杂指令遵循：让模型学会执行复杂的多步骤任务
3. 领域术语内化：让模型理解特定领域的专业术语和缩略语
4. 推理效率要求高：不需要额外的检索步骤

三、综合方案
现代企业级应用通常采用RAG + 微调的混合方案：
1. 微调优化模型的行为模式（输出风格、指令遵循）
2. RAG提供实时和领域特定的知识支持
3. 两者互补：微调让模型更擅长利用检索结果，RAG让模型的知识不过时

四、性能与成本考量
- RAG：推理成本高（需额外检索步骤），但维护成本低
- 微调：推理成本低（单次推理），但训练和维护成本高"""
    },
    # ── 21-30: 深度学习模型 ──
    {
        "title": "卷积神经网络CNN详解",
        "content": """卷积神经网络（Convolutional Neural Network，CNN）是深度学习在计算机视觉领域最重要的架构之一。
CNN的设计灵感来源于生物视觉皮层中神经元对局部感受野的响应模式。

CNN的三大核心设计思想：
1. 局部连接（Local Connectivity）：每个神经元只连接输入的一小片区域（感受野）
2. 权重共享（Weight Sharing）：同一个卷积核在整个输入上滑动，共享参数
3. 平移不变性（Translation Invariance）：无论特征出现在图像的哪个位置，都能被检测到

CNN的层次结构：
输入层 → 卷积层 → 激活函数 → 池化层 → ... → 全连接层 → 输出层

卷积层的核心参数：
- 卷积核大小（Kernel Size）：通常3x3或5x5，小卷积核堆叠效果更好
- 步长（Stride）：卷积核滑动的步长
- 填充（Padding）：在输入边界补充像素，控制输出尺寸
- 通道数（Filters）：卷积核的数量，决定输出特征图的深度
- 输出尺寸公式：output_size = (input_size - kernel_size + 2*padding) / stride + 1

池化层的类型：
- 最大池化（Max Pooling）：取区域内的最大值，保留最显著特征
- 平均池化（Average Pooling）：取区域内的平均值，保留背景信息
- 全局平均池化（Global Average Pooling）：对整个特征图取平均，替代全连接层

经典CNN架构演进时间线：
1998年 - LeNet-5（Yann LeCun）：手写数字识别，7层网络
2012年 - AlexNet：ImageNet冠军，深度学习的开端
2014年 - VGGNet：使用3x3小卷积核堆叠，19层
2014年 - GoogLeNet/Inception：引入Inception模块，多尺度特征提取
2015年 - ResNet：残差连接解决退化问题，152层
2017年 - DenseNet：密集连接，特征复用
2019年 - EfficientNet：神经架构搜索（NAS）找到最优缩放比例"""
    },
    {
        "title": "循环神经网络RNN与LSTM",
        "content": """循环神经网络（Recurrent Neural Network，RNN）是一类用于处理序列数据的神经网络架构，
其核心特点是网络中的循环连接使得信息可以在时间步之间传递，形成"记忆"。

标准RNN的局限性：
1. 梯度消失（Vanishing Gradient）：长序列中早期时间步的梯度趋近于零，无法学习长距离依赖
2. 梯度爆炸（Exploding Gradient）：梯度过大导致训练不稳定
3. 长期依赖学习困难：理论上RNN可以处理任意长度，实际上只能记住最近几步的信息

LSTM（长短期记忆网络）的解决方案：
LSTM通过引入门控机制（Gating Mechanism）来控制信息的流动。

LSTM的核心组件：
1. 遗忘门（Forget Gate）：决定从细胞状态中丢弃哪些旧信息
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. 输入门（Input Gate）：决定将哪些新信息存入细胞状态
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)

3. 细胞状态更新：
   C_t = f_t * C_{t-1} + i_t * C̃_t

4. 输出门（Output Gate）：决定从细胞状态中输出哪些信息
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)

GRU（门控循环单元）：
GRU是LSTM的简化版本，将遗忘门和输入门合并为"更新门"：
- 参数更少，训练更快
- 在许多任务上效果与LSTM相当
- 适合数据量较少的场景

应用场景：
- 自然语言处理：文本分类、序列标注、机器翻译
- 时间序列预测：股票价格、天气预测
- 语音识别：声学模型"""
    },
    {
        "title": "生成对抗网络GAN",
        "content": """生成对抗网络（Generative Adversarial Network，GAN）由Ian Goodfellow在2014年提出，
是一种通过对抗训练生成高质量合成数据的深度学习框架。

GAN的基本架构：
1. 生成器（Generator, G）：输入随机噪声z，输出合成数据G(z)，目标是生成以假乱真的数据
2. 判别器（Discriminator, D）：输入真实数据x或合成数据G(z)，输出判断为"真"的概率

对抗训练的数学表示：
min_G max_D V(D,G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]

训练过程：
1. 固定生成器，训练判别器：提升判别器区分真假的能力
2. 固定判别器，训练生成器：提升生成器欺骗判别器的能力
3. 交替迭代，最终达到纳什均衡

GAN的主要变体：
1. DCGAN（深度卷积GAN）：将CNN引入GAN架构，稳定训练
2. Conditional GAN（条件GAN）：通过条件信息（如类别标签）控制生成内容
3. CycleGAN：实现无配对数据的图像风格迁移（如照片→梵高风格）
4. StyleGAN：通过风格向量控制生成图像的不同层次的视觉特征，如面部朝向、发型等。StyleGAN已成为生成高质量人脸的标杆方案
5. Pix2Pix：基于配对数据的图像翻译（如草图→照片）

GAN的训练挑战：
- 模式坍塌（Mode Collapse）：生成器只生成少数几种模式
- 训练不稳定：生成器和判别器需要精细平衡
- 评估困难：缺少客观的生成质量评估指标
- 梯度消失：判别器太强导致生成器梯度为零"""
    },
    {
        "title": "扩散模型Diffusion Models",
        "content": """扩散模型（Diffusion Models）是近年来生成模型领域的重大突破，在图像生成质量上超越了GAN。
Stable Diffusion、DALL-E 2、Midjourney等知名图像生成系统都基于扩散模型。

扩散模型的核心原理：
1. 前向扩散过程（Forward Process）：逐步向数据添加高斯噪声，直到数据变为纯噪声
2. 反向去噪过程（Reverse Process）：学习从噪声中逐步恢复原始数据

数学表示：
- 前向过程：q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t*I)
- 反向过程：p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t^2*I)

扩散模型的优势：
1. 生成质量：图片质量通常优于GAN，细节丰富
2. 训练稳定：不需要对抗训练，损失函数简单
3. 多样性好：不容易出现模式坍塌
4. 可控性强：通过引导（Guidance）技术控制生成方向

DDPM（去噪扩散概率模型）的关键技术：
- 使用U-Net作为噪声预测网络
- 时间步嵌入（Timestep Embedding）：告知网络当前的噪声水平
- 注意力机制：提升长距离依赖的建模能力

Stable Diffusion的优化：
- 在潜空间（Latent Space）而非像素空间进行扩散，大幅降低计算量
- 使用VAE将图像压缩到低维潜空间
- 通过文本编码器（CLIP Text Encoder）实现文本到图像的生成

应用场景：
- 文生图：DALL-E 3、Stable Diffusion、Midjourney
- 图像编辑：Inpainting（填补缺失区域）、Outpainting（扩展画面）
- 超分辨率：将低分辨率图像提升为高分辨率
- 3D生成：从文本描述生成3D模型"""
    },
    {
        "title": "PyTorch框架基础",
        "content": """PyTorch是由Meta AI开发的开源深度学习框架，因其动态计算图和Python风格编程而广受欢迎。

PyTorch的核心组件：

1. 张量（Tensor）：
   类似于NumPy的ndarray，但支持GPU加速和自动微分。
   - 创建：torch.tensor(), torch.zeros(), torch.randn()
   - 操作：加减乘除、矩阵运算、索引切片
   - 设备迁移：.to('cuda') / .to('cpu')

2. 自动微分（Autograd）：
   自动计算神经网络中所有参数的梯度。
   - 追踪计算图：设置requires_grad=True
   - 反向传播：loss.backward()
   - 梯度清零：optimizer.zero_grad()

3. 神经网络模块（torch.nn）：
   - nn.Linear：全连接层
   - nn.Conv2d：2D卷积层
   - nn.LSTM、nn.GRU：循环网络层
   - nn.Transformer：Transformer层
   - nn.BatchNorm2d：批归一化
   - nn.Dropout：随机失活
   - 自定义模块：继承nn.Module，实现__init__和forward方法

4. 优化器（torch.optim）：
   - optim.SGD：随机梯度下降
   - optim.Adam：自适应矩估计
   - optim.AdamW：带权重衰减的Adam
   学习率调度器：StepLR、CosineAnnealingLR、ReduceLROnPlateau

5. 数据加载（torch.utils.data）：
   - Dataset：自定义数据集类
   - DataLoader：批量加载、打乱、多进程预取

PyTorch训练标准流程：
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch.x)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()

PyTorch vs TensorFlow：
- PyTorch：动态图、Python风格、研究社区首选
- TensorFlow/Keras：静态图为主、部署生态更成熟、工业界广泛使用"""
    },
    # ── 31-40: AI应用场景 ──
    {
        "title": "AI在医疗领域的应用",
        "content": """人工智能正在深刻变革医疗健康行业，从疾病诊断到药物研发，AI技术在提高医疗效率和准确性方面展现出巨大潜力。

一、医学影像分析
1. 放射影像辅助诊断：
   - 肺结节检测：CNN模型在CT图像中检测肺结节，敏感度超过放射科医生平均水平
   - 骨折检测：在X光片上自动标记骨折位置
   - 脑出血识别：在CT头部扫描中快速识别颅内出血

2. 病理图像分析：
   - 癌细胞检测：在病理切片中自动识别和分类癌细胞
   - 组织分割：识别不同组织类型，辅助量化分析

3. 眼底图像分析：
   - 糖尿病视网膜病变筛查
   - 青光眼早期检测

二、药物研发
1. 靶点发现：利用知识图谱和NLP技术挖掘药物-疾病-基因关联
2. 分子设计：生成模型设计具有特定性质的新分子结构
3. 虚拟筛选：在数百万化合物库中快速筛选潜在药物
4. 临床试验优化：预测患者招募成功率，优化试验设计

三、临床决策支持
1. 疾病风险预测：基于电子健康记录预测疾病风险
2. 治疗方案推荐：根据患者特征推荐个体化治疗方案
3. 用药安全：检测潜在的药物相互作用和禁忌症"""

# 病例分析辅助：
# AI系统分析海量病例数据，辅助医生进行鉴别诊断
- 罕见病识别：利用知识库帮助识别少见疾病
- 疾病进展预测：基于时序数据预测疾病发展趋势
    },
    {
        "title": "AI在金融领域的应用",
        "content": """金融行业是人工智能技术应用最深入、价值最显著的领域之一。AI在风控、交易、客服等多个金融场景中发挥着关键作用。

一、风险管理和欺诈检测
1. 信用评估：
   - 基于机器学习的信用评分模型：使用XGBoost、LightGBM等算法
   - 替代数据源分析：利用社交网络、手机使用行为等非传统数据
   - 小微企业信用评估：结合供应链数据和企业关联图谱

2. 欺诈检测：
   - 实时交易监控：使用异常检测算法识别可疑交易
   - 反洗钱（AML）：知识图谱识别复杂的洗钱网络
   - 保险欺诈：NLP分析理赔文本，发现欺诈模式

3. 市场风险：
   - VaR（风险价值）预测：深度学习预测极端市场风险
   - 压力测试：模拟极端市场条件下的投资组合表现

二、量化交易和投资策略
1. 价格预测：LSTM、Transformer预测股票价格趋势
2. 因子挖掘：自动发现有效的量化因子
3. 高频交易：强化学习优化订单执行策略
4. 另类数据分析：NLP分析新闻、社交媒体情绪预测市场

三、智能客服和金融助手
1. 智能客服：基于大语言模型的7×24小时客服
2. 投资顾问：个性化投资建议和资产配置
3. 金融知识库：RAG系统提供实时金融政策和产品信息

四、监管科技（RegTech）
1. 合规检查：自动审核交易是否符合监管要求
2. 文档处理：NLP自动提取和分析大量法律文书
3. 监管报告：自动生成符合监管要求的报告"""
    },
    {
        "title": "AI在教育领域的应用",
        "content": """人工智能技术正在重新定义教育的形态，从个性化学习到智能评估，AI为教育带来了前所未有的变革机遇。

一、个性化学习
1. 自适应学习系统：
   - 知识图谱驱动的学习路径规划
   - 基于学生能力模型的动态难度调整
   - 个性化练习题推荐

2. 智能辅导系统（Intelligent Tutoring System, ITS）：
   - 一对一虚拟辅导：模拟人类教师的引导式提问
   - 错误诊断：分析学生解题过程中的常见错误模式
   - 提示生成：在学生遇到困难时提供适当程度的提示

3. 学习内容生成：
   - 自动生成练习题和答案解析
   - 根据学生兴趣定制阅读材料
   - 多语言教材翻译和本地化

二、自动评估和反馈
1. 作文自动评分：
   - 基于BERT等预训练模型的作文评分
   - 多维度评估：内容、结构、语法、词汇
   - 详细的修改建议和写作指导

2. 编程作业评估：
   - 代码正确性验证
   - 代码风格和质量分析
   - 抄袭检测

3. 口语评估：
   - 发音准确度评估
   - 流利度和语调分析

三、教育管理和辅助
1. 学情分析：基于学习数据预测学生表现和辍学风险
2. 智能排课：优化课程时间安排和教室资源分配
3. 招生辅助：基于历史数据预测招生趋势
4. 特殊教育辅助：AI辅助听障、视障学生的学习"""
    },
    {
        "title": "AI在自动驾驶中的应用",
        "content": """自动驾驶是人工智能最具挑战性和影响力的应用领域之一，涉及感知、规划、控制等多个技术模块的协同工作。

一、环境感知
1. 视觉感知：
   - 目标检测：YOLO、Faster R-CNN检测车辆、行人、交通标志
   - 语义分割：对道路场景进行像素级分类（车道、行人、建筑物）
   - 深度估计：从单目或双目图像估计距离

2. 多传感器融合：
   - 摄像头 + 激光雷达（LiDAR）+ 毫米波雷达
   - 卡尔曼滤波、粒子滤波进行传感器数据融合
   - BEV（鸟瞰视角）感知：将多传感器数据统一到俯视视角

3. 预测模块：
   - 轨迹预测：预测周围车辆和行人的未来运动轨迹
   - 意图预测：判断其他交通参与者即将采取的行为
   - 社会交互建模：考虑交通参与者之间的相互影响

二、规划与控制
1. 路径规划：
   - 全局规划：A*算法、RRT（快速随机树）
   - 局部规划：考虑动态障碍物的实时避障
   - 行为规划：变道、超车、让行等高层决策

2. 运动控制：
   - PID控制器：经典的反馈控制方法
   - 模型预测控制（MPC）：考虑未来多个时间步的优化控制
   - 端到端控制：直接从传感器输入到控制指令

三、自动驾驶等级
- L1：辅助驾驶（定速巡航或车道保持，单一功能）
- L2：部分自动化（同时进行定速巡航和车道保持）
- L3：有条件自动化（特定条件下系统完全负责，但驾驶员需随时接管）
- L4：高度自动化（限定区域内无需人类干预）
- L5：完全自动化（任何道路和环境条件下无需人类驾驶）"""
    },
    {
        "title": "推荐系统技术详解",
        "content": """推荐系统是现代互联网应用的核心技术，从电商、视频到社交，推荐系统决定了用户体验和商业价值。

一、推荐系统的核心方法

1. 协同过滤（Collaborative Filtering）：
   - 基于用户的协同过滤：找相似用户，推荐他们喜欢的物品
   - 基于物品的协同过滤：找相似物品，推荐给喜欢相关物品的用户
   - 矩阵分解（MF）：将用户-物品交互矩阵分解为用户隐向量和物品隐向量
   - SVD++、ALS等优化算法

2. 基于内容的推荐（Content-Based Filtering）：
   - 物品特征提取：利用NLP、CV技术提取物品的文本和视觉特征
   - 用户画像构建：基于用户历史行为构建兴趣模型
   - 相似度匹配：计算候选物品与用户兴趣的匹配度

3. 深度学习推荐：
   - Wide & Deep（Google）：记忆能力+泛化能力
   - DeepFM：结合因子分解机和深度神经网络
   - DIN（阿里）：基于注意力机制的深度兴趣网络
   - YouTube DNN：两阶段（召回+排序）推荐架构

二、推荐系统的工程架构
1. 召回层（Recall）：从百万/亿级候选集中筛选数百个候选
   - 多路召回：协同过滤、向量检索、热门召回、地理召回
   - 向量化召回：利用Embedding进行近似最近邻搜索

2. 排序层（Ranking）：对数百候选进行精细排序
   - 精排模型：LR、GBDT、DeepFM、DCN等
   - 多目标优化：同时优化点击率、转化率、停留时长等

3. 重排层（Re-ranking）：考虑多样性、新颖性、业务规则
   - MMR算法：最大化边际相关性
   - 打散策略：避免同类内容连续出现

三、推荐系统的评估指标
- 离线指标：AUC、NDCG、召回率、精确率
- 在线指标：CTR（点击率）、CVR（转化率）、用户停留时长
- 多样性指标：类别覆盖度、内容新颖度、惊喜度"""
    },
    # ── 41-50: AI工具与实践 ──
    {
        "title": "Python机器学习工具链",
        "content": """Python是人工智能和机器学习领域最主流的编程语言，拥有丰富的生态系统和工具链。

一、核心数据处理库
1. NumPy：
   - 高性能多维数组操作
   - 线性代数运算、傅里叶变换
   - 随机数生成
   - 广播机制（Broadcasting）：不同形状数组间的运算

2. Pandas：
   - DataFrame和Series数据结构
   - 数据清洗、缺失值处理
   - 分组聚合、透视表
   - 时间序列分析

3. Matplotlib/Seaborn：
   - 折线图、散点图、柱状图、热力图
   - 统计可视化
   - 交互式图表（Plotly、Bokeh）

二、机器学习框架
1. Scikit-learn：
   - 统一API设计：fit/predict/transform
   - 丰富的算法库：分类、回归、聚类、降维
   - 模型选择：交叉验证、网格搜索
   - 预处理：标准化、编码、特征选择

2. XGBoost/LightGBM/CatBoost：
   - 梯度提升树算法
   - 处理表格数据的首选方案
   - 支持GPU加速、分布式训练

三、深度学习框架
1. PyTorch + PyTorch Lightning：研究灵活 + 工程规范
2. Hugging Face Transformers：预训练模型生态
3. LangChain/LlamaIndex：LLM应用开发框架

四、MLOps工具
1. MLflow：实验跟踪、模型注册、部署
2. Weights & Biases：实验可视化和协作
3. DVC：数据和模型版本控制
4. FastAPI：模型推理API服务
5. Docker/Kubernetes：容器化部署和编排

五、中文NLP工具
1. jieba分词：中文分词
2. HanLP：多任务中文NLP工具包
3. paddlenlp：百度飞桨NLP工具包"""
    },
    {
        "title": "LangChain框架详解",
        "content": """LangChain是一个用于构建大语言模型（LLM）应用的开发框架，由Harrison Chase于2022年创建，
提供了标准化的组件和接口来简化LLM应用的开发。

LangChain的核心模块：

1. Model I/O（模型输入输出）
   - LLM/ChatModel：统一的大语言模型调用接口
   - Prompt Template：模板化提示词管理，支持Few-shot示例
   - Output Parser：结构化输出解析

2. Retrieval（检索）
   - Document Loaders：支持150+种文档格式的加载器
   - Text Splitters：多种文本分割策略
   - Embedding Models：Embedding模型接口
   - Vector Stores：Chroma、Pinecone、FAISS等向量库的统一接口
   - Retrievers：检索器抽象

3. Chains（链）
   - LLMChain：基础的提示词+LLM链
   - SequentialChain：多个链的顺序组合
   - RetrievalQA：检索增强的问答链
   - ConversationalRetrievalChain：对话式检索链

4. Agents（智能体）
   - 工具定义：搜索、计算、API调用等
   - 推理循环：思考→行动→观察→思考
   - ReAct Agent：推理+行动的循环范式
   - Plan-and-Execute：先规划再执行

5. Memory（记忆）
   - ConversationBufferMemory：存储完整对话历史
   - ConversationSummaryMemory：存储对话摘要
   - VectorStoreRetrieverMemory：基于向量的记忆检索

LangChain的典型RAG应用架构：
Document Loader → Text Splitter → Embedding Model → Vector Store
                                                        ↑
User Query → Retriever → Retrieved Docs → Prompt Template → LLM → Answer

LangChain的局限性：
- 抽象层级过多，调试困难
- 版本更新频繁，API不稳定
- 复杂场景下灵活性不足
- 对生产环境支持不够完善

替代框架：
- LlamaIndex：侧重于数据索引和检索
- Haystack：面向生产的NLP框架
- DSPy：声明式的LLM编程框架"""
    },
    {
        "title": "Ollama本地部署指南",
        "content": """Ollama是一个轻量级的本地大语言模型运行框架，支持在个人电脑上轻松部署和运行各种开源LLM。
尤其适合隐私敏感场景、离线环境和开发测试。

一、Ollama的核心特性
1. 一键安装：支持Windows、macOS、Linux
2. 模型管理：ollama pull/push/list/rm管理模型
3. REST API：兼容OpenAI格式的HTTP API
4. 多模态支持：部分模型支持图像输入
5. GPU加速：自动检测并利用NVIDIA/AMD GPU
6. 模型量化：支持多种量化级别减少显存占用

二、常用模型
1. 通用对话：qwen2.5（通义千问）、llama3（Meta）、mistral
2. 代码生成：codellama、deepseek-coder-v2、qwen2.5-coder
3. Embedding：nomic-embed-text、bge-m3、mxbai-embed-large
4. 视觉理解：llava、minicpm-v、gemma3
5. 推理增强：deepseek-r1（思考链）

三、API调用示例
1. 对话补全（Chat Completions）：
POST /api/chat
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "你好"}],
  "stream": true
}

2. 文本向量化（Embeddings）：
POST /api/embeddings
{
  "model": "nomic-embed-text",
  "prompt": "需要向量化的文本"
}

四、性能优化建议
1. 选择合适的量化级别：Q4_K_M在质量和速度间平衡最佳
2. 设置合理的上下文长度：num_ctx参数
3. 调整GPU层数：num_gpu参数控制GPU卸载比例
4. 并发控制：OLLAMA_NUM_PARALLEL环境变量
5. 使用环境变量管理配置：OLLAMA_HOST、OLLAMA_MODELS等

五、生产部署考量
1. 配合反向代理（Nginx/Caddy）暴露API
2. 使用Docker容器化部署
3. 结合Cloudflare Tunnel实现安全远程访问
4. 监控显存使用和请求延迟
5. 定期更新Ollama和模型版本"""
    },
    {
        "title": "Chroma向量数据库实践",
        "content": """Chroma是一个开源的嵌入式向量数据库，专为AI应用设计，是构建RAG系统最便捷的向量存储方案之一。

一、Chroma的安装与初始化
安装：pip install chromadb

基本使用流程：
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_knowledge")

二、核心API操作
1. 添加数据：
collection.add(
    documents=["文档内容1", "文档内容2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

2. 查询数据：
results = collection.query(
    query_texts=["查询文本"],
    n_results=5,
    where={"source": "doc1"},  # 元数据过滤
    include=["documents", "metadatas", "distances"]
)

3. 更新和删除：
collection.update(ids=["id1"], documents=["新内容"])
collection.delete(ids=["id1"])
collection.delete(where={"source": "doc1"})

4. 获取数据：
collection.get(ids=["id1"])
collection.get(where={"source": "doc1"})
collection.count()  # 获取数据量

三、Chroma的索引和距离度量
1. 索引类型：默认HNSW（Hierarchical Navigable Small World）
   - HNSW参数：M（连接数）、ef_construction（构建搜索宽度）
   - 可通过metadata调整：{"hnsw:space": "cosine"}

2. 距离度量：
   - cosine：余弦距离（默认，适合语义搜索）
   - l2：欧氏距离
   - ip：内积（适合归一化向量）

四、Chroma的局限性
1. 单机部署，不支持分布式
2. 数据量大时（>100万）性能下降
3. 没有原生的全文搜索支持
4. 备份和迁移功能不完善

五、Chroma与RAG的集成模式
1. 文档索引：Document → Chunks → Embed → Chroma
2. 查询检索：Query → Embed → Chroma.query → Results
3. 元数据过滤：结合文档来源、时间等元数据过滤
4. 增量更新：支持动态添加和删除文档向量"""
    },
    {
        "title": "FastAPI后端开发实战",
        "content": """FastAPI是一个现代、高性能的Python Web框架，基于标准Python类型提示构建API，
是构建AI应用后端服务的理想选择。

一、FastAPI的核心优势
1. 高性能：基于Starlette和Pydantic，性能媲美Node.js和Go
2. 自动文档：自动生成Swagger UI和ReDoc交互式API文档
3. 类型安全：利用Python类型提示进行数据验证和序列化
4. 异步支持：原生支持async/await异步编程
5. 标准化：完全兼容OpenAPI和JSON Schema标准
6. 依赖注入：灵活的依赖注入系统

二、FastAPI应用基础结构
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/items")
async def create_item(item: Item):
    return {"item": item}

# 路由分组
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1")

@router.get("/users")
async def get_users():
    return {"users": []}

app.include_router(router)

三、请求处理和验证
1. 路径参数：
@app.get("/items/{item_id}")
def get_item(item_id: int): ...

2. 查询参数：
@app.get("/items/")
def list_items(skip: int = 0, limit: int = 10): ...

3. 请求体（Pydantic模型）：
from pydantic import BaseModel
class Item(BaseModel):
    name: str
    price: float
    description: str | None = None

4. 文件上传：
from fastapi import UploadFile, File
@app.post("/upload")
async def upload(file: UploadFile = File(...)): ...

四、流式响应（适用于LLM场景）
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat():
    async def generate():
        for chunk in llm_stream():
            yield f"data: {json.dumps(chunk)}\\n\\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

五、部署与优化
1. Uvicorn运行：
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

2. Gunicorn + Uvicorn（多进程）：
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

3. CORS配置：
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

4. 速率限制：slowapi库
5. 日志记录：Python logging + structlog"""
    },
    {
        "title": "Docker容器化部署",
        "content": """Docker是一个开源的容器化平台，通过将应用及其依赖打包到轻量级容器中，实现一致的环境和便捷的部署。

一、Docker核心概念
1. 镜像（Image）：应用及其运行环境的只读模板
2. 容器（Container）：镜像的运行实例，相互隔离
3. Dockerfile：构建镜像的指令文件
4. 仓库（Registry）：存储和分发镜像（Docker Hub、私有仓库）
5. 卷（Volume）：持久化数据和共享数据

二、AI应用Dockerfile示例
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

多阶段构建优化镜像体积：
FROM python:3.11 AS builder
RUN pip install --user package
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local

三、Docker Compose编排多服务
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  chroma:
    image: chromadb/chroma
    volumes:
      - chroma_data:/chroma/chroma

  app:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - ollama
      - chroma
    environment:
      - OLLAMA_HOST=http://ollama:11434

四、容器化最佳实践
1. 使用.dockerignore排除不需要的文件
2. 避免以root用户运行容器
3. 使用健康检查确保服务可用
4. 合理设置资源限制（CPU/内存）
5. 使用多阶段构建减小镜像体积
6. 敏感信息通过环境变量或secrets管理
7. 镜像标签使用语义化版本"""
    },
    {
        "title": "Git版本控制与协作",
        "content": """Git是分布式版本控制系统，是软件开发和AI项目管理的基础工具。

一、Git核心工作流程
1. 工作区 → 暂存区 → 本地仓库 → 远程仓库
2. 基本命令：
   git init          # 初始化仓库
   git add .         # 添加到暂存区
   git commit -m ""  # 提交到本地仓库
   git push          # 推送到远程仓库
   git pull          # 拉取远程更新

二、分支管理策略
1. Git Flow：
   - main：生产环境代码
   - develop：开发主线
   - feature/*：功能分支
   - release/*：发布分支
   - hotfix/*：紧急修复分支

2. GitHub Flow（简化版）：
   - main分支始终保持可部署状态
   - 功能开发在feature分支进行
   - 通过Pull Request合并

3. 提交信息规范（Conventional Commits）：
   feat: 新功能
   fix: 修复bug
   docs: 文档更新
   refactor: 代码重构
   test: 测试相关
   chore: 构建/工具变更

三、协作工作流
1. Pull Request（PR）流程：
   - 创建feature分支 → 开发 → 推送 → 创建PR → 代码审查 → 合并

2. 代码审查要点：
   - 代码逻辑正确性
   - 代码风格和可读性
   - 测试覆盖率
   - 安全性问题

3. 冲突解决：
   - git merge vs git rebase
   - 手动解决冲突 → git add → git commit

四、AI项目的Git最佳实践
1. 使用.gitignore忽略大型模型文件
2. 使用Git LFS管理大文件（数据集、模型权重）
3. 使用DVC（Data Version Control）管理数据和模型版本
4. 构建脚本自动化：GitHub Actions、GitLab CI
5. 敏感信息管理：绝不提交API密钥，使用环境变量"""
    },
    {
        "title": "AI项目开发流程",
        "content": """规范化的AI项目开发流程是保障项目成功交付的关键，涵盖了从需求分析到部署运维的完整生命周期。

一、项目启动阶段
1. 问题定义：明确业务目标和AI技术的适用性
2. 可行性评估：
   - 数据可用性：是否有足够的高质量数据
   - 技术可行性：现有技术能否解决问题
   - 投入产出比：AI投入与预期收益的评估
3. 团队组建：数据工程师、算法工程师、后端工程师、前端工程师
4. 技术选型：确定技术栈和基础设施

二、数据准备阶段
1. 数据采集：从数据库、API、爬虫、传感器等来源收集数据
2. 数据探索（EDA）：
   - 统计分析：均值、方差、分布
   - 可视化：散点图、分布图、相关性热图
   - 异常检测：离群点、缺失值
3. 数据清洗：
   - 处理缺失值：删除、填充均值/中位数、模型预测填充
   - 处理异常值：截断、变换
   - 数据去重
4. 数据标注：人工标注、半自动标注、众包标注
5. 数据增强：增加数据多样性，提升模型泛化能力

三、模型开发阶段
1. 基线建立：使用简单模型（如逻辑回归）建立性能基线
2. 模型选型：根据任务选择合适算法
3. 特征工程：构建、选择和变换特征
4. 模型训练：超参数调优，交叉验证
5. 模型评估：多维度评估指标
6. 错误分析：分析模型失败案例，指导优化方向

四、部署运维阶段
1. 模型优化：量化、剪枝、蒸馏
2. 推理服务：FastAPI/Flask + Docker容器化
3. 监控告警：模型性能下降、数据漂移检测
4. 模型更新：定期重训练、在线学习
5. A/B测试：对比新旧模型效果"""
    },
    {
        "title": "大模型微调技术综述",
        "content": """大语言模型的微调（Fine-tuning）技术使预训练模型能够适应特定领域和任务需求，
是构建垂直领域AI应用的重要手段。

一、全量微调（Full Fine-tuning）
更新模型的所有参数，效果最好但资源消耗最大。
- 适用：有充足的GPU资源和大量标注数据
- 缺点：计算成本高、容易过拟合、灾难性遗忘

二、参数高效微调（PEFT）
只更新模型的一小部分参数，大幅降低计算和存储需求。

1. LoRA（低秩适应）：
   在预训练权重旁添加低秩矩阵：W' = W + BA
   - rank r通常为8-64
   - 可减少99%的可训练参数
   - 支持多任务切换（切换不同的LoRA权重）

2. QLoRA：
   LoRA + 4-bit量化，极大降低显存需求
   - 在单张24GB GPU上可微调65B参数模型

3. Adapter：
   在Transformer层间插入小型可训练网络，原模型参数冻结

4. Prefix Tuning：
   在输入前添加可训练的连续前缀向量

5. P-Tuning v2：
   在每一层添加可训练的提示向量

三、指令微调（Instruction Tuning）
使用（指令，回答）对训练模型遵循指令的能力
- 数据构建：Self-Instruct、Evol-Instruct自动生成指令数据
- 典型数据集：Alpaca、Dolly、OpenOrca

四、RLHF（人类反馈强化学习）
1. 收集人类偏好数据：对同一提示的不同回答进行排序
2. 训练奖励模型：学习预测人类偏好
3. PPO优化：使用奖励模型通过PPO算法优化策略

DPO（直接偏好优化）：
- 简化RLHF流程，直接从偏好对中优化
- 损失函数：L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]

五、微调实践建议
1. 数据质量 > 数据数量：少量高质量数据效果优于大量低质量数据
2. 从PEFT开始：先用LoRA验证效果，再决定是否全量微调
3. 防止灾难性遗忘：混合通用数据和领域数据
4. 评估多样性：在多个维度上评估微调效果"""
    },
]


def generate_documents():
    """生成50篇知识文档到指定目录"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    for i, doc in enumerate(DOCUMENTS, 1):
        filename = f"{i:02d}_{doc['title'].replace('/', '_')}.txt"
        filepath = OUTPUT_DIR / filename
        filepath.write_text(doc["content"], encoding="utf-8")
        generated.append({"index": i, "title": doc["title"], "file": filename})
        print(f"  [{i:02d}/50] {doc['title']}")

    print(f"\n✅ 已生成 {len(generated)} 篇文档到: {OUTPUT_DIR}")
    print(f"   可直接用于知识库上传测试")

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps({"total": len(generated), "documents": generated}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"   文档清单: {manifest_path}")


if __name__ == "__main__":
    generate_documents()
