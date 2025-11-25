import sqlite3
import json
import os
# 从 config 文件导入 DB_NAME
from config import DB_NAME

#数据库连接函数
def get_db_connection():
    try:
        # 连接到本地文件数据库，如果文件不存在会自动创建
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row  #增加代码可读性
        return conn 
    except sqlite3.Error as e:
        print(f"数据库连接失败: {e}")
        return None

def initialize_database():
    conn = get_db_connection()
    if conn is None: return #若conn对象为空值，直接返回。
    
    try:
        cursor = conn.cursor()
        
        # 1. 创建 sim_parameters 表，多行字符串
        cursor.execute(''' 
            CREATE TABLE IF NOT EXISTS sim_parameters (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                K_topics INTEGER,
                V_size INTEGER,
                N_docs INTEGER,
                alpha_param REAL,
                sim_date TEXT,
                final_similarity_score REAL
            )
        ''')
        
        # 2. 创建 documents_data 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents_data (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                simulated_text TEXT,
                true_theta_vector TEXT,
                FOREIGN KEY(run_id) REFERENCES sim_parameters(run_id)
            )
        ''')
        
        # 3. 创建 analysis_results 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                run_id INTEGER,
                predicted_theta_vector TEXT,
                cosine_similarity REAL,
                FOREIGN KEY(run_id) REFERENCES sim_parameters(run_id)
            )
        ''')
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"初始化数据库表失败: {e}")
    finally:
        conn.close()

#记录参数函数
def record_simulation_parameters(params):
    conn = get_db_connection()
    if conn is None: return None

    run_id = None
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO sim_parameters (K_topics, V_size, N_docs, alpha_param, sim_date) VALUES (?, ?, ?, ?, ?)"
        cursor.execute(sql, params)
        run_id = cursor.lastrowid #获取ID
        conn.commit()
    except sqlite3.Error as e:
        print(f"参数记录失败: {e}")
        conn.rollback()
    finally:
        conn.close()
    return run_id

#批量插入文档
def bulk_insert_documents(documents_data):
    conn = get_db_connection()
    if conn is None: return False
    
    insert_success = False
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO documents_data (run_id, simulated_text, true_theta_vector) VALUES (?, ?, ?)"
        cursor.executemany(sql, documents_data)
        conn.commit()
        print(f"{len(documents_data)} 条文档数据批量入库成功。")
        insert_success = True
            
    except sqlite3.Error as e:
        print(f"文档批量入库失败: {e}")
        conn.rollback()
    finally:
        conn.close()
    return insert_success

#数据提取函数
def fetch_documents_for_analysis(run_id):
    conn = get_db_connection()
    if conn is None: return None
    
    data_list = []
    try:
        cursor = conn.cursor()
        sql = "SELECT doc_id, simulated_text, true_theta_vector FROM documents_data WHERE run_id = ?"
        cursor.execute(sql, (run_id,))
        
        results = cursor.fetchall()
        
        for row in results:
            true_theta = json.loads(row['true_theta_vector']) 
            data_list.append({
                'doc_id': row['doc_id'],
                'text': row['simulated_text'],
                'true_theta': true_theta
            })
            
        print(f"从数据库成功提取 {len(data_list)} 条文档数据。")
            
    except Exception as e:
        print(f"数据提取失败: {e}")
        data_list = None
    finally:
        conn.close()
    return data_list

#批量插入分析结果
def bulk_insert_analysis_results(results_list):
    conn = get_db_connection()
    if conn is None: return False
    
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO analysis_results (doc_id, run_id, predicted_theta_vector, cosine_similarity) VALUES (?, ?, ?, ?)"
        cursor.executemany(sql, results_list)
        conn.commit()
        print(f"{len(results_list)} 条分析结果批量入库成功。")
            
    except sqlite3.Error as e:
        print(f"分析结果批量入库失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
    return True

#更新模拟结果
def update_simulation_results(run_id, final_similarity_score):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "UPDATE sim_parameters SET final_similarity_score = ? WHERE run_id = ?"
        cursor.execute(sql, (final_similarity_score, run_id))
        conn.commit()
        conn.close()
        print(f"[成功] run_id {run_id} 的最终平均相似度分数已更新。")
        return True
    except Exception as e:
        print(f"[失败] 更新模拟结果时出错: {e}")
        return False