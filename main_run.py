from config import K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE, vocabulary, DOC_LENGTH
# 1. 导入更新: 替换为更通用的 update_simulation_results
from db_manager import record_simulation_parameters, bulk_insert_documents, fetch_documents_for_analysis, bulk_insert_analysis_results, update_simulation_results 
# 修正导入名称：从 stat_sim 导入模拟与分析函数
from stat_sim import create_phi_matrix, generate_documents, train_and_predict_lda, calculate_cosine_similarity 

import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation as LDA 

def main():
    print("--- LDA统计模拟项目启动 ---")
    params = (K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE)
    print("1. 尝试连接数据库并记录实验参数...")
    current_run_id = record_simulation_parameters(params)
    
    if current_run_id is None:
        print("[失败] 无法获取 run_id，请检查数据库连接配置。")
        return
    
    print(f"[成功] 实验参数记录成功。本次 run_id: {current_run_id}")

    print("\n2. 统计模拟数据生成阶段启动 ---")
    
    # 2.1 创建 Phi 矩阵 (主题-词语分布)
    print("   - 正在创建高区分度的 Phi 矩阵...")
    phi_matrix = create_phi_matrix(K_TOPICS, V_SIZE)
    
    # 2.2 生成 N 篇文档和真实 theta_true 向量
    generated_data = generate_documents(
        phi_matrix, 
        vocabulary, 
        current_run_id, 
        ALPHA_PARAM, 
        N_DOCS, 
        DOC_LENGTH
    )
    
    # 2.3 批量入库文档数据
    print("\n3. 正在批量入库文档数据...")
    success = False
    if generated_data:
        success = bulk_insert_documents(generated_data)
        
    if success:
        print("[成功] 3000 条文档数据入库成功。")
    else:
        print("[失败] 数据入库失败或数据为空，请检查日志。")
        return # 入库失败则终止后续分析

    print("\n4. 模型训练与核心验证阶段启动 ---")
      
    # 4.1.1 从数据库提取数据 (获取模拟文本和 theta_true)
    print("   - 正在从数据库提取数据...")
    analysis_data = fetch_documents_for_analysis(current_run_id)
    
    if not analysis_data:
        print("[失败] 未能从数据库提取数据，终止分析。")
        return
        
    # 4.1.2 预处理: 构建 DTM (文档-词语矩阵)
    print("   - 正在构建 DTM...")
    texts = [d['text'] for d in analysis_data]
    
    vectorizer = CountVectorizer(vocabulary=vocabulary) 
    dtm = vectorizer.fit_transform(texts)
    
    print(f"[成功] DTM 构建成功。矩阵形状: {dtm.shape}")
    
    # 4.2 模型训练与推断 (只返回 theta_pred)
    theta_pred_matrix = train_and_predict_lda(dtm, dtm, K_TOPICS)
    
    analysis_results = []
    
    # 4.3 余弦相似度量化验证
    print("   - 正在执行余弦相似度量化验证...")
    for i, doc_data in enumerate(analysis_data):
        # 1. 获取真实 theta 和预测 theta
        true_theta = doc_data['true_theta']          # 列表形式 (从数据库加载，shape (5,))
        
        # 2. 核心修正点：使用索引 [i] 确保只提取当前文档的预测向量
        pred_theta_vector = theta_pred_matrix[i] 
        pred_theta = pred_theta_vector.tolist()      # numpy 数组转为列表 (shape (5,))
        
        # 3. 计算余弦相似度 (两个 (5,) 向量进行对比)
        similarity = calculate_cosine_similarity(true_theta, pred_theta)
        
        # 4. 组装数据元组：(doc_id, run_id, predicted_theta_vector, cosine_similarity)
        result_tuple = (
            doc_data['doc_id'],
            current_run_id,
            json.dumps(pred_theta), 
            similarity
        )
        analysis_results.append(result_tuple)
        
    # 4.4 批量插入 analysis_results 表
    bulk_insert_analysis_results(analysis_results)
    print("[成功] 3000 条分析结果入库成功。")
    
    # 4.5 最终输出和总结
    # 从分析结果列表中提取所有相似度分数，计算平均值
    avg_similarity = np.mean([r[3] for r in analysis_results]) 
    
    # 5. 更新 sim_parameters 表中的总分数
    update_simulation_results(current_run_id, avg_similarity) 
    
    print(f"\n--- **项目最终验证结果** ---")
    print(f"**平均余弦相似度 (Cosine Sim. $\\theta_{{true}}$ vs $\\theta_{{pred}}$):** {avg_similarity:.4f}")
    print("\n**项目全部完成！** 所有模拟数据和分析结果已保存到数据库。")


if __name__ == "__main__":
    main()