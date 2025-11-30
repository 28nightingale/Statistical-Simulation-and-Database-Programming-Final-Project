import numpy as np
from numpy.random import dirichlet
from numpy.random import choice as np_choice
from numpy.linalg import norm 
import json
from sklearn.decomposition import LatentDirichletAllocation as LDA

# 步骤 1:构建 Phi 矩阵 (词语-主题分布)
def create_phi_matrix(K, V):
    """构建 K x V 的 Phi 矩阵，保证主题区分度。矩阵元素为对应概率"""
    
    num_words_per_topic = V // K  
    phi_matrix = np.zeros((K, V))
    
    high_prob_share = 0.90  
    low_prob_share = 0.10  
    
    #保证概率归一化
    HIGH_PROB = high_prob_share / num_words_per_topic 
    LOW_PROB = low_prob_share / (V - num_words_per_topic) 

    for k in range(K): 
        start_index = k * num_words_per_topic
        end_index = (k + 1) * num_words_per_topic
        
        # 1. 设置主题词的高概率 (90% 的概率质量)
        phi_matrix[k, start_index:end_index] = HIGH_PROB
        
        # 2. 设置非主题词的低概率 (剩余 10% 的概率质量)
        if start_index > 0:
            phi_matrix[k, 0:start_index] = LOW_PROB
            
        if end_index < V:
            phi_matrix[k, end_index:V] = LOW_PROB

    # 再次归一化，确保浮点数不影响
    phi_matrix = phi_matrix / phi_matrix.sum(axis=1, keepdims=True)
    
    return phi_matrix

# 步骤 2: 文档生成函数 (使用 Phi 矩阵)

def generate_documents(phi_matrix, vocabulary, run_id, alpha_param, n_docs, doc_length):
    """
    根据 LDA 的生成原理，生成 N_DOCS 篇文档和真实 theta 向量。
    返回格式化后的数据列表，用于批量入库。
    """
    all_documents_data = []
    K = phi_matrix.shape[0]
    V = phi_matrix.shape[1]
    
    # 1. 核心统计抽样: 从 Dirichlet 分布中抽取真实主题比例向量 (theta_true)
    true_theta_vectors = dirichlet([alpha_param] * K, size=n_docs)
    
    for i in range(n_docs):
        theta_true = true_theta_vectors[i]
        document_words = []
        
        # 2. 循环生成文档中的每个词语
        for _ in range(doc_length):
            # a) 抽主题: 根据文档的 theta_true 比例，从 K 个主题中抽取一个主题索引
            topic_index = np_choice(np.arange(K), p=theta_true)
            
            # b) 抽词语: 根据被抽中的主题的 Phi 概率，从 V 个词汇中抽取一个词语索引
            word_index = np_choice(np.arange(V), p=phi_matrix[topic_index, :])
            word = vocabulary[word_index]
            
            document_words.append(word)
            
        # 3. 组装数据，准备入库
        simulated_text = " ".join(document_words)
        
        # 将 numpy 数组转换为列表，再转换为 JSON 字符串，SQLite只能存储文本、整数等，不能存Numpy数组
        theta_json = json.dumps(theta_true.tolist())
        
        # 数据元组格式: (run_id, simulated_text, theta_json)
        all_documents_data.append((
            run_id,
            simulated_text,
            theta_json
        ))
        
        if (i + 1) % 500 == 0:
            print(f"    已生成 {i + 1} / {n_docs} 篇文档")

    print("文档生成完成。")
    return all_documents_data
    
# 步骤 3: 模型训练与推
def train_and_predict_lda(dtm, data_dtm, K_topics):    
    # 1. 配置LDA 模型
    lda_model = LDA(n_components=K_topics, 
                    max_iter=10, #最大迭代次数
                    learning_method='batch', #模型学习方法
                    random_state=42) #随机种子，保证实验结果可重复可比较。
    
    # 2. 训练模型 
    print("正在训练 LDA 模型 (K=5)")
    lda_model.fit(dtm)
    
    # 3. 推断主题分布 (theta_pred)
    print("正在推断文档主题分布 (theta_pred)")
    theta_pred_matrix = lda_model.transform(data_dtm)
    
    print("模型训练完成。")

    return theta_pred_matrix

# 步骤 4: 余弦相似度计算函数

def calculate_cosine_similarity(vec_a, vec_b):
    """计算两个向量之间的余弦相似度"""
    # 确保输入是 numpy 数组
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    # 1. 计算点积 (A . B)
    # 此处假设 vec_a 和 vec_b 都是形状 (5,) 的一维向量，
    dot_product = np.dot(vec_a, vec_b)
    
    # 2. 计算模长 (L2 范数)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    
    # 防止除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    similarity = dot_product / (norm_a * norm_b)
    return similarity