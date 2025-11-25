from config import K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE, vocabulary, DOC_LENGTH
# 【修改点1】导入 initialize_database
from db_manager import record_simulation_parameters, bulk_insert_documents, fetch_documents_for_analysis, bulk_insert_analysis_results, update_simulation_results, initialize_database
from stat_sim import create_phi_matrix, generate_documents, train_and_predict_lda, calculate_cosine_similarity 
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer 

def main():
    # 【修改点2】确保数据库表存在
    print("--- 正在初始化环境 ---")
    initialize_database()

    print("--- LDA统计模拟项目启动 ---")
    params = (K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, str(SIM_DATE))
    print("1. 记录实验参数...")
    current_run_id = record_simulation_parameters(params)
    
    if current_run_id is None:
        print("[失败] 无法获取 run_id")
        return
    
    print(f"[成功] Run ID: {current_run_id}")
    print("\n2. 模拟数据生成...")
    phi_matrix = create_phi_matrix(K_TOPICS, V_SIZE)
    generated_data = generate_documents(phi_matrix, vocabulary, current_run_id, ALPHA_PARAM, N_DOCS, DOC_LENGTH)
    
    print("\n3. 批量入库...")
    if generated_data:
        bulk_insert_documents(generated_data)
    else:
        return

    print("\n4. 验证阶段...")
    analysis_data = fetch_documents_for_analysis(current_run_id)
    texts = [d['text'] for d in analysis_data]
    vectorizer = CountVectorizer(vocabulary=vocabulary) 
    dtm = vectorizer.fit_transform(texts)
    
    theta_pred_matrix = train_and_predict_lda(dtm, dtm, K_TOPICS)
    
    analysis_results = []
    print("   - 计算相似度...")
    for i, doc_data in enumerate(analysis_data):
        true_theta = doc_data['true_theta']     
        pred_theta = theta_pred_matrix[i].tolist()
        similarity = calculate_cosine_similarity(true_theta, pred_theta)
        
        result_tuple = (doc_data['doc_id'], current_run_id, json.dumps(pred_theta), similarity)
        analysis_results.append(result_tuple)
        
    bulk_insert_analysis_results(analysis_results)
    avg_similarity = np.mean([r[3] for r in analysis_results]) 
    update_simulation_results(current_run_id, avg_similarity) 
    
    print(f"\n--- **项目最终验证结果** ---")
    print(f"**平均余弦相似度:** {avg_similarity:.4f}")

if __name__ == "__main__":
    main()