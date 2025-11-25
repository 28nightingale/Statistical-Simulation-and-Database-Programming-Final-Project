import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import scrolledtext
import threading
import sys
import json
import numpy as np

# --- 导入项目核心模块 ---
from config import K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE, vocabulary, DOC_LENGTH, DB_NAME
# 【修改点1】导入 initialize_database
from db_manager import record_simulation_parameters, bulk_insert_documents, fetch_documents_for_analysis, bulk_insert_analysis_results, update_simulation_results, initialize_database
from stat_sim import create_phi_matrix, generate_documents, train_and_predict_lda, calculate_cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.after(0, self._write, str)

    def _write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end") 
        self.widget.configure(state="disabled")

    def flush(self):
        pass

class LDAApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("LDA 统计模拟与验证系统")
        self.geometry("1000x700")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.create_menu()
        self.create_toolbar()
        self.create_main_panels()
        self.stdout_backup = sys.stdout

    def create_menu(self):
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="退出", command=self.quit_app)
        menu_bar.add_cascade(label="文件(F)", menu=file_menu)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="关于项目", command=self.show_about)
        menu_bar.add_cascade(label="帮助(H)", menu=help_menu)
        self.config(menu=menu_bar)

    def create_toolbar(self):
        toolbar = tk.Frame(self, bg="#f0f0f0", bd=1, relief="raised")
        toolbar.pack(side="top", fill="x")
        btn_style = {"padx": 10, "pady": 5, "bg": "#e1e1e1"}
        self.btn_run = tk.Button(toolbar, text="▶ 开始完整模拟", command=self.start_simulation_thread, fg="green", font=("Arial", 10, "bold"), **btn_style)
        self.btn_run.pack(side="left", padx=5, pady=5)
        tk.Button(toolbar, text="清空日志", command=self.clear_log, **btn_style).pack(side="left", padx=5, pady=5)
        tk.Button(toolbar, text="退出系统", command=self.quit_app, bg="#ffcccc").pack(side="right", padx=5, pady=5)

    def create_main_panels(self):
        self.paned_window = tk.PanedWindow(self, orient="horizontal", sashrelief="raised")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)

        # --- 左侧：参数与状态面板 ---
        left_frame = tk.Frame(self.paned_window, width=280, bg="white", relief="sunken", bd=1)
        left_frame.pack_propagate(False)
        
        lbl_title = tk.Label(left_frame, text="当前实验配置", font=("Arial", 12, "bold"), bg="#0078d7", fg="white", pady=5)
        lbl_title.pack(fill="x")

        # 参数展示
        params = [
            ("数据库模式", "SQLite (本地文件)"),
            ("数据库文件", DB_NAME),
            ("主题数 (K)", K_TOPICS),
            ("词汇量 (V)", V_SIZE),
            ("文档数 (N)", N_DOCS),
            ("Alpha 参数", ALPHA_PARAM),
            ("文档长度", DOC_LENGTH),
            ("模拟时间", SIM_DATE.strftime("%Y-%m-%d %H:%M"))
        ]
        
        param_text_frame = tk.Frame(left_frame, bg="white", padx=10, pady=10)
        param_text_frame.pack(fill="both")
        
        for i, (key, val) in enumerate(params):
            tk.Label(param_text_frame, text=f"{key}:", font=("Arial", 10, "bold"), bg="white", anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(param_text_frame, text=f"{val}", font=("Arial", 10), bg="white", fg="blue", anchor="w").grid(row=i, column=1, sticky="w", pady=2)

        tk.Label(left_frame, text="最终结果", font=("Arial", 12, "bold"), bg="#28a745", fg="white", pady=5).pack(fill="x", pady=(20, 0))
        self.lbl_result_title = tk.Label(left_frame, text="平均余弦相似度", font=("Arial", 10), bg="white", pady=5)
        self.lbl_result_title.pack()
        self.lbl_result_score = tk.Label(left_frame, text="--", font=("Arial", 24, "bold"), fg="red", bg="white")
        self.lbl_result_score.pack()
        self.paned_window.add(left_frame)

        # --- 右侧：日志 ---
        right_frame = tk.Frame(self.paned_window, bg="white", relief="sunken", bd=1)
        tk.Label(right_frame, text="系统运行日志", bg="#f0f0f0", anchor="w", padx=5).pack(fill="x")
        self.log_text = scrolledtext.ScrolledText(right_frame, state='disabled', font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("stdout", foreground="black")
        self.paned_window.add(right_frame)
        sys.stdout = TextRedirector(self.log_text, "stdout")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.configure(state="disabled")

    def start_simulation_thread(self):
        self.btn_run.config(state="disabled", text="正在运行...")
        self.lbl_result_score.config(text="计算中...")
        thread = threading.Thread(target=self.run_full_simulation)
        thread.daemon = True
        thread.start()

    def run_full_simulation(self):
        try:
            print("--- LDA统计模拟项目启动  ---")
            print(f"正在使用数据库文件: {DB_NAME}")
            
            # 其余逻辑保持不变
            print("1. 记录实验参数...")
            params_tuple = (K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, str(SIM_DATE))
            current_run_id = record_simulation_parameters(params_tuple)
            
            if current_run_id is None:
                print("[错误] 无法获取 run_id。")
                self.reset_ui_state()
                return

            print(f"[成功] Run ID: {current_run_id}")
            print("\n2. 模拟数据生成...")
            phi_matrix = create_phi_matrix(K_TOPICS, V_SIZE)
            generated_data = generate_documents(phi_matrix, vocabulary, current_run_id, ALPHA_PARAM, N_DOCS, DOC_LENGTH)
            
            print("\n3. 批量入库...")
            success = bulk_insert_documents(generated_data)
            if not success:
                print("[错误] 入库失败。")
                self.reset_ui_state()
                return
            
            print("\n4. 模型训练与验证...")
            analysis_data = fetch_documents_for_analysis(current_run_id)
            texts = [d['text'] for d in analysis_data]
            vectorizer = CountVectorizer(vocabulary=vocabulary)
            dtm = vectorizer.fit_transform(texts)
            theta_pred_matrix = train_and_predict_lda(dtm, dtm, K_TOPICS)
            
            print("   - 计算相似度...")
            analysis_results = []
            for i, doc_data in enumerate(analysis_data):
                true_theta = doc_data['true_theta']
                pred_theta_vector = theta_pred_matrix[i]
                similarity = calculate_cosine_similarity(true_theta, pred_theta_vector.tolist())
                result_tuple = (doc_data['doc_id'], current_run_id, json.dumps(pred_theta_vector.tolist()), similarity)
                analysis_results.append(result_tuple)
            
            bulk_insert_analysis_results(analysis_results)
            avg_similarity = np.mean([r[3] for r in analysis_results])
            update_simulation_results(current_run_id, avg_similarity)
            
            print(f"\n=== 最终结果: {avg_similarity:.4f} ===")
            self.after(0, self.update_final_result, avg_similarity)
            self.after(0, lambda: messagebox.showinfo("成功", f"模拟完成！\n相似度: {avg_similarity:.4f}"))

        except Exception as e:
            print(f"\n[错误] {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.reset_ui_state()

    def update_final_result(self, score):
        self.lbl_result_score.config(text=f"{score:.4f}")

    def reset_ui_state(self):
        self.after(0, lambda: self.btn_run.config(state="normal", text="▶ 开始完整模拟"))

    def show_about(self):
        messagebox.showinfo("关于", "LDA 统计模拟演示系统 (Portable)")

    def quit_app(self):
        sys.stdout = self.stdout_backup
        self.destroy()

if __name__ == "__main__":
    # 【修改点2】程序启动时，自动初始化数据库表结构
    initialize_database()
    
    app = LDAApp()
    app.mainloop()