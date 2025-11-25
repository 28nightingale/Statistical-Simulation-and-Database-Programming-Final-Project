import sqlite3
import json
import os

from config import DB_NAME

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"数据库连接失败：{e}")
        return None

def record_simulation_parameters(params):
    conn = get_db_connection()
    if conn is None: return
    run_id = None
    try:
        cursor = conn.cursor()
        sql = 'INSERT INTO sim_parameters (K_topics,V_size,N_docs,alpha_param,sim_date) VALUES(?,?,?,?,?)'
        cursor.execute(sql,params)
        run_id = cursor.lastrowid
        conn.commit()
    except sqlite3.Error as e:
        print(f"参数记录失败: {e}")
        conn.rollback
    finally:
        conn.commit
    return run_id

def main():
    

