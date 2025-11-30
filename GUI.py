import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import json
import numpy as np

from config import K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE, vocabulary, DOC_LENGTH, DB_NAME
from db_manager import record_simulation_parameters, bulk_insert_documents, fetch_documents_for_analysis, bulk_insert_analysis_results, update_simulation_results, initialize_database
from stat_sim import create_phi_matrix, generate_documents, train_and_predict_lda, calculate_cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class SimpleLDAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LDA 统计模拟助手 (简化版)")
        self.geometry("800x600") #窗口大小
        self.style = ttk.Style()
        self.style.theme_use('clam')

        #初始化数据库
        initialize_database()

        self.create_menu()
        self.create_toolbar()
        self.create_main_panels()

    def create_menu(self):#创建顶部菜单栏
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="退出", command=self.quit)
        menu_bar.add_cascade(label="文件(F)", menu=file_menu)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="关于项目", command=lambda: messagebox.showinfo("关于", "LDA 统计模拟演示系统"))
        menu_bar.add_cascade(label="帮助(H)", menu=help_menu)
        self.config(menu=menu_bar)

    def create_toolbar(self):
        toolbar = tk.Frame(self, bg="#f0f0f0", bd=1, relief="raised")
        toolbar.pack(side="top", fill="x")
        btn_style = {"padx": 10, "pady": 5, "bg": "#e1e1e1"}
        
        self.btn_run = tk.Button(toolbar, text="▶ 运行 LDA 模拟", 
                                 command=self.run_simulation_and_show_result, 
                                 fg="green", font=("Arial", 10, "bold"), **btn_style)
        self.btn_run.pack(side="left", padx=5, pady=5)
        
        tk.Button(toolbar, text="退出系统", command=self.quit, bg="#ffcccc").pack(side="right", padx=5, pady=5)

    def create_main_panels(self):
        self.paned_window = tk.PanedWindow(self, orient="horizontal", sashrelief="raised")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)

        #左侧：参数面板
        left_frame = tk.Frame(self.paned_window, width=280, bg="white", relief="sunken", bd=1)
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="当前实验配置", font=("Arial", 12, "bold"), bg="#0078d7", fg="white", pady=5).pack(fill="x")
        
        params = [
            ("数据库文件", DB_NAME),
            ("主题数 (K)", K_TOPICS),
            ("文档数 (N)", N_DOCS),
            ("Alpha 参数", ALPHA_PARAM),
            ("文档长度", DOC_LENGTH),
        ]
        param_text_frame = tk.Frame(left_frame, bg="white", padx=10, pady=10)
        param_text_frame.pack(fill="both")
        for i, (key, val) in enumerate(params):
            tk.Label(param_text_frame, text=f"{key}:", font=("Arial", 10, "bold"), bg="white", anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(param_text_frame, text=f"{val}", font=("Arial", 10), bg="white", fg="blue", anchor="w").grid(row=i, column=1, sticky="w", pady=2)

        tk.Label(left_frame, text="最终结果", font=("Arial", 12, "bold"), bg="#28a745", fg="white", pady=5).pack(fill="x", pady=(20, 0))
        tk.Label(left_frame, text="平均余弦相似度", font=("Arial", 10), bg="white", pady=5).pack()
        self.lbl_result_score = tk.Label(left_frame, text="--", font=("Arial", 24, "bold"), fg="red", bg="white")
        self.lbl_result_score.pack()
        self.paned_window.add(left_frame)

        #右侧：简单状态区 
        right_panel = tk.Frame(self.paned_window, bg="#f0f0f0")
        self.status_label = tk.Label(right_panel, text="点击 '运行 LDA 模拟' 开始验证", font=("Arial", 14), bg="#f0f0f0")
        self.status_label.pack(expand=True)
        self.paned_window.add(right_panel)

    def run_simulation_and_show_result(self):
        try:
            self.status_label.config(text="正在运行中，请等待...", fg="orange")
            self.lbl_result_score.config(text="计算中...")
            self.btn_run.config(state="disabled")

            # 1. 记录实验参数
            params_tuple = (K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, str(SIM_DATE))
            current_run_id = record_simulation_parameters(params_tuple)
            
            if current_run_id is None:
                messagebox.showerror("错误", "无法获取 run_id，请检查数据库连接。")
                return

            # 2. 模拟数据生成
            phi_matrix = create_phi_matrix(K_TOPICS, V_SIZE)
            generated_data = generate_documents(phi_matrix, vocabulary, current_run_id, ALPHA_PARAM, N_DOCS, DOC_LENGTH)
            
            # 3. 批量入库
            success = bulk_insert_documents(generated_data)
            if not success:
                messagebox.showerror("错误", "文档入库失败。")
                return
            
            # 4. 模型训练与验证
            analysis_data = fetch_documents_for_analysis(current_run_id)
            texts = [d['text'] for d in analysis_data]
            vectorizer = CountVectorizer(vocabulary=vocabulary)
            dtm = vectorizer.fit_transform(texts)
            theta_pred_matrix = train_and_predict_lda(dtm, dtm, K_TOPICS)
            
            # 5. 计算相似度并组装结果
            analysis_results = []
            for i, doc_data in enumerate(analysis_data):
                true_theta = doc_data['true_theta']
                # 注意：这里直接将 NumPy 数组转换为列表
                pred_theta_vector = theta_pred_matrix[i] 
                similarity = calculate_cosine_similarity(true_theta, pred_theta_vector.tolist())
                
                result_tuple = (doc_data['doc_id'], current_run_id, json.dumps(pred_theta_vector.tolist()), similarity)
                analysis_results.append(result_tuple)
            
            # 6. 存档并更新总分
            bulk_insert_analysis_results(analysis_results)
            avg_similarity = np.mean([r[3] for r in analysis_results])
            update_simulation_results(current_run_id, avg_similarity)
            
            # 7. 更新 UI
            self.lbl_result_score.config(text=f"{avg_similarity:.4f}")
            self.status_label.config(text=f"模拟完成，平均相似度: {avg_similarity:.4f}", fg="green")
            messagebox.showinfo("成功", f"模拟完成！\n平均余弦相似度: {avg_similarity:.4f}")

        except Exception as e:
            self.status_label.config(text=f"运行失败，错误: {str(e)[:50]}...", fg="red")
            self.lbl_result_score.config(text="错误")
            messagebox.showerror("运行错误", f"运行过程中发生错误:\n{e}")
            
        finally:
            self.btn_run.config(state="normal") # 恢复按钮状态

if __name__ == "__main__":
    app = SimpleLDAApp()
    app.mainloop()