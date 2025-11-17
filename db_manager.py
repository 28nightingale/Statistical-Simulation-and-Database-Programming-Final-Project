import pymysql
import json
# 从 config 文件导入所有必要的常量
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE

#数据库连接函数
def get_db_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except pymysql.MySQLError as e:
        # 在 db_manager 中不打印连接成功，只处理失败，成功日志放在 main_run 中
        print(f"数据库连接失败，错误信息: {e.args[1]}")
        return None
    
#记录参数函数
def record_simulation_parameters(params):
    """将实验参数记录到 sim_parameters 表中，并返回 run_id"""
    conn = get_db_connection()
    if conn is None: return None

    run_id = None
    try:
        with conn.cursor() as cursor:
            # SQL 语句：
            sql = "INSERT INTO sim_parameters (K_topics, V_size, N_docs, alpha_param, sim_date) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql, params)
            run_id = cursor.lastrowid
            conn.commit()
    except pymysql.MySQLError as e:
        print(f"参数记录失败: {e}")
        conn.rollback()
    finally:
        conn.close()
    return run_id

#记录参数函数
def bulk_insert_documents(documents_data):
    """批量插入生成的文档数据到 documents_data 表"""
    conn = get_db_connection()
    if conn is None: return False
    
    insert_success = False
    try:
        with conn.cursor() as cursor:
            # SQL 语句：注意这里对应 documents_data 表的三个字段
            # (run_id, simulated_text, true_theta_vector)
            sql = "INSERT INTO documents_data (run_id, simulated_text, true_theta_vector) VALUES (%s, %s, %s)"
            
            # *** 核心操作：使用 executemany 实现批量插入 ***
            cursor.executemany(sql, documents_data)
            
            conn.commit()
            print(f"{len(documents_data)} 条文档数据批量入库成功。")
            insert_success = True
            
    except pymysql.MySQLError as e:
        print(f"文档批量入库失败: {e}")
        conn.rollback()
    finally:
        conn.close()
    return insert_success

#数据提取函数
def fetch_documents_for_analysis(run_id):
    """从 documents_data 表中提取所有文档数据和真实的 theta 向量"""
    conn = get_db_connection()
    if conn is None: return None
    
    data_list = []
    try:
        with conn.cursor() as cursor:
            # 查询 doc_id, simulated_text 和 true_theta_vector
            sql = "SELECT doc_id, simulated_text, true_theta_vector FROM documents_data WHERE run_id = %s"
            cursor.execute(sql, (run_id,))
            
            results = cursor.fetchall()
            
            for row in results:
                # 注意：从 MySQL JSON 字段取出的数据通常是字符串，需要 json.loads 解析
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
# db_manager.py (新增函数部分)

# --- 批量插入分析结果 (Task 4.3.2) ---
def bulk_insert_analysis_results(results_list):
    """批量插入模型分析结果到 analysis_results 表"""
    conn = get_db_connection()
    if conn is None: return False
    
    try:
        with conn.cursor() as cursor:
            # SQL 语句：对应 analysis_results 表的字段
            # (doc_id, run_id, predicted_theta_vector, cosine_similarity)
            sql = "INSERT INTO analysis_results (doc_id, run_id, predicted_theta_vector, cosine_similarity) VALUES (%s, %s, %s, %s)"
            
            # 使用 executemany 进行批量插入
            cursor.executemany(sql, results_list)
            
            conn.commit()
            print(f"{len(results_list)} 条分析结果批量入库成功。")
            
    except pymysql.MySQLError as e:
        print(f"分析结果批量入库失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
    return True
    
# (确保您已导入所需的库，如 mysql.connector 或您正在使用的数据库连接库)
# from . import get_db_connection  # 或其他连接方式

# ... 其他 db_manager 函数 ...

def update_simulation_results(run_id, final_similarity_score):
    """
    更新 sim_parameters 表中指定 run_id 的最终验证分数（平均余弦相似度）。
    :param run_id: 本次模拟实验的唯一 ID。
    :param final_similarity_score: 计算出的平均余弦相似度分数 (float)。
    :return: True/False
    """
    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        # 假设 sim_parameters 表中有一个字段名为 'final_similarity_score'
        # 如果您使用的是不同的字段名，请在此处修改 SQL 语句
        sql = """
        UPDATE sim_parameters 
        SET final_similarity_score = %s 
        WHERE run_id = %s
        """
        cursor.execute(sql, (final_similarity_score, run_id))
        db.commit()
        cursor.close()
        db.close()
        print(f"[成功] run_id {run_id} 的最终平均相似度分数已更新。")
        return True
    except Exception as e:
        print(f"[失败] 更新模拟结果时出错: {e}")
        return False