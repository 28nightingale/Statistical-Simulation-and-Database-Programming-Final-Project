import datetime
#数据库连接常量
DB_HOST = 'localhost'
DB_USER = 'root' 
DB_PASSWORD = '20220823qjy' 
DB_NAME = 'lda_sim_db' 

#实验常量
K_TOPICS = 5        # 主题数 K
V_SIZE = 1000       # 词汇表大小 V
N_DOCS = 3000       # 模拟文档总数 N
ALPHA_PARAM = 0.5   # Dirichlet 分布的超参数 alpha，控制文档主题分布稀疏性
SIM_DATE = datetime.datetime.now()# 模拟运行时间，区分不同实验
DOC_LENGTH = 200    # 文档平均长度

#词汇表 (Vocabulary)
vocabulary = []
num_of_words_per_topic = 200
topics = ['Movies_A', 'Sports_B', 'Finance_C', 'Life_D', 'Literature_E']

# 生成1000个词
for topic in topics:
    for i in range(1, num_of_words_per_topic + 1):
        word = topic + str(i).zfill(3) # 统一用三位数
        vocabulary.append(word)