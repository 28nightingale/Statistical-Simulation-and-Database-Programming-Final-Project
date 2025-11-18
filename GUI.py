import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import scrolledtext
import threading
import sys
import json
import numpy as np

# --- 导入项目核心模块 ---
from config import K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE, vocabulary, DOC_LENGTH, DB_HOST, DB_USER, DB_NAME
from db_manager import record_simulation_parameters, bulk_insert_documents, fetch_documents_for_analysis, bulk_insert_analysis_results, update_simulation_results
from stat_sim import create_phi_matrix, generate_documents, train_and_predict_lda, calculate_cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class TextRedirector(object):
    """
    一个辅助类，用于将 print() 的输出重定向到 GUI 的文本框中
    """
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        # 使用 after 方法确保在主线程中更新 UI，防止多线程冲突
        self.widget.after(0, self._write, str)

    def _write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end") # 自动滚动到底部
        self.widget.configure(state="disabled")

    def flush(self):
        pass

class LDAApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("LDA 统计模拟与验证系统 - 控制台")
        self.geometry("1000x700")
        
        # 设置整体样式
        self.style = ttk.Style()
        self.style.theme_use('clam') # 使用较现代的主题

        # 1. 创建菜单栏
        self.create_menu()

        # 2. 创建顶部工具栏
        self.create_toolbar()

        # 3. 创建主界面布局 (左右分栏)
        self.create_main_panels()

        # 恢复标准输出的备份
        self.stdout_backup = sys.stdout

    def create_menu(self):
        menu_bar = tk.Menu(self)

        # 文件菜单
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="退出", command=self.quit_app)
        menu_bar.add_cascade(label="文件(F)", menu=file_menu)

        # 帮助菜单
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="关于项目", command=self.show_about)
        menu_bar.add_cascade(label="帮助(H)", menu=help_menu)

        self.config(menu=menu_bar)

    def create_toolbar(self):
        toolbar = tk.Frame(self, bg="#f0f0f0", bd=1, relief="raised")
        toolbar.pack(side="top", fill="x")

        # 定义按钮风格
        btn_style = {"padx": 10, "pady": 5, "bg": "#e1e1e1"}

        # 一键运行按钮
        self.btn_run = tk.Button(toolbar, text="▶ 开始完整模拟", command=self.start_simulation_thread, fg="green", font=("Arial", 10, "bold"), **btn_style)
        self.btn_run.pack(side="left", padx=5, pady=5)

        # 清空日志按钮
        tk.Button(toolbar, text="清空日志", command=self.clear_log, **btn_style).pack(side="left", padx=5, pady=5)

        # 退出按钮
        tk.Button(toolbar, text="退出系统", command=self.quit_app, bg="#ffcccc").pack(side="right", padx=5, pady=5)

    def create_main_panels(self):
        # 使用 PanedWindow 实现左右拖动调整大小
        self.paned_window = tk.PanedWindow(self, orient="horizontal", sashrelief="raised")
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)

        # --- 左侧：参数与状态面板 ---
        left_frame = tk.Frame(self.paned_window, width=280, bg="white", relief="sunken", bd=1)
        left_frame.pack_propagate(False) # 防止被内容撑大
        
        # 参数显示区
        lbl_title = tk.Label(left_frame, text="当前实验配置 (Config)", font=("Arial", 12, "bold"), bg="#0078d7", fg="white", pady=5)
        lbl_title.pack(fill="x")

        # 使用 Treeview 展示参数
        self.param_tree = ttk.Treeview(left_frame, columns=("Value"), show="headings", height=10)
        self.param_tree.heading("Value", text="参数值")
        self.param_tree.column("Value", anchor="w")
        
        # 插入配置参数
        params = [
            ("数据库主机", DB_HOST),
            ("数据库用户", DB_USER),
            ("数据库名", DB_NAME),
            ("主题数 (K)", K_TOPICS),
            ("词汇量 (V)", V_SIZE),
            ("文档数 (N)", N_DOCS),
            ("Alpha 参数", ALPHA_PARAM),
            ("文档长度", DOC_LENGTH),
            ("模拟时间", SIM_DATE)
        ]
        
        # 使用 Label 显示参数列表（比 Treeview 更直观一点）
        param_text_frame = tk.Frame(left_frame, bg="white", padx=10, pady=10)
        param_text_frame.pack(fill="both")
        
        for i, (key, val) in enumerate(params):
            tk.Label(param_text_frame, text=f"{key}:", font=("Arial", 10, "bold"), bg="white", anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(param_text_frame, text=f"{val}", font=("Arial", 10), bg="white", fg="blue", anchor="w").grid(row=i, column=1, sticky="w", pady=2)

        # 结果展示区
        tk.Label(left_frame, text="最终结果", font=("Arial", 12, "bold"), bg="#28a745", fg="white", pady=5).pack(fill="x", pady=(20, 0))
        
        self.lbl_result_title = tk.Label(left_frame, text="平均余弦相似度", font=("Arial", 10), bg="white", pady=5)
        self.lbl_result_title.pack()
        
        self.lbl_result_score = tk.Label(left_frame, text="--", font=("Arial", 24, "bold"), fg="red", bg="white")
        self.lbl_result_score.pack()

        self.paned_window.add(left_frame)

        # --- 右侧：日志输出面板 ---
        right_frame = tk.Frame(self.paned_window, bg="white", relief="sunken", bd=1)
        
        tk.Label(right_frame, text="系统运行日志", bg="#f0f0f0", anchor="w", padx=5).pack(fill="x")
        
        # 滚动文本框
        self.log_text = scrolledtext.ScrolledText(right_frame, state='disabled', font=("Consolas", 10))
        self.log_text.pack(fill="both", expand=True)

        # 配置日志颜色标签
        self.log_text.tag_config("stdout", foreground="black")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("success", foreground="green")

        self.paned_window.add(right_frame)

        # 重定向 print 输出
        sys.stdout = TextRedirector(self.log_text, "stdout")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.configure(state="disabled")

    def start_simulation_thread(self):
        """在一个单独的线程中运行模拟，避免卡死界面"""
        self.btn_run.config(state="disabled", text="正在运行...")
        self.lbl_result_score.config(text="计算中...")
        
        # 创建线程
        thread = threading.Thread(target=self.run_full_simulation)
        thread.daemon = True # 设为守护线程，主程序退出时它也会退出
        thread.start()

    def run_full_simulation(self):
        """这是原本 main.py 中的核心逻辑"""
        try:
            print("--- LDA统计模拟项目启动 ---")
            print("1. 尝试连接数据库并记录实验参数...")
            
            params_tuple = (K_TOPICS, V_SIZE, N_DOCS, ALPHA_PARAM, SIM_DATE)
            current_run_id = record_simulation_parameters(params_tuple)
            
            if current_run_id is None:
                print("[错误] 无法获取 run_id，请检查配置。")
                self.reset_ui_state()
                return

            print(f"[成功] 实验参数记录成功。Run ID: {current_run_id}")
            print("\n2. 统计模拟数据生成阶段启动...")
            
            print("   - 正在创建高区分度的 Phi 矩阵...")
            phi_matrix = create_phi_matrix(K_TOPICS, V_SIZE)
            
            print(f"   - 正在生成 {N_DOCS} 篇文档...")
            generated_data = generate_documents(phi_matrix, vocabulary, current_run_id, ALPHA_PARAM, N_DOCS, DOC_LENGTH)
            
            print("\n3. 正在批量入库文档数据...")
            success = False
            if generated_data:
                success = bulk_insert_documents(generated_data)
            
            if not success:
                print("[错误] 数据入库失败。")
                self.reset_ui_state()
                return
            
            print("[成功] 文档数据入库完毕。")
            print("\n4. 模型训练与核心验证阶段启动...")
            
            print("   - 正在从数据库提取数据...")
            analysis_data = fetch_documents_for_analysis(current_run_id)
            
            if not analysis_data:
                print("[错误] 提取数据为空。")
                self.reset_ui_state()
                return

            print("   - 正在构建 DTM (CountVectorizer)...")
            texts = [d['text'] for d in analysis_data]
            vectorizer = CountVectorizer(vocabulary=vocabulary)
            dtm = vectorizer.fit_transform(texts)
            
            print("   - 正在训练 LDA 模型并推断 (这可能需要几秒钟)...")
            theta_pred_matrix = train_and_predict_lda(dtm, dtm, K_TOPICS)
            
            print("   - 正在执行余弦相似度验证...")
            analysis_results = []
            for i, doc_data in enumerate(analysis_data):
                true_theta = doc_data['true_theta']
                pred_theta_vector = theta_pred_matrix[i]
                pred_theta = pred_theta_vector.tolist()
                
                similarity = calculate_cosine_similarity(true_theta, pred_theta)
                
                result_tuple = (
                    doc_data['doc_id'],
                    current_run_id,
                    json.dumps(pred_theta),
                    similarity
                )
                analysis_results.append(result_tuple)
            
            bulk_insert_analysis_results(analysis_results)
            
            avg_similarity = np.mean([r[3] for r in analysis_results])
            update_simulation_results(current_run_id, avg_similarity)
            
            print(f"\n=== 项目最终验证结果 ===")
            print(f"Run ID: {current_run_id}")
            print(f"平均余弦相似度: {avg_similarity:.4f}")
            print("===========================")
            
            # 更新 UI 结果
            self.after(0, self.update_final_result, avg_similarity)
            self.after(0, lambda: messagebox.showinfo("成功", f"模拟完成！\n平均余弦相似度: {avg_similarity:.4f}"))

        except Exception as e:
            print(f"\n[致命错误] {str(e)}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror("错误", f"运行过程中发生错误:\n{str(e)}"))
        
        finally:
            self.reset_ui_state()

    def update_final_result(self, score):
        self.lbl_result_score.config(text=f"{score:.4f}")

    def reset_ui_state(self):
        self.after(0, lambda: self.btn_run.config(state="normal", text="▶ 开始完整模拟"))

    def show_about(self):
        messagebox.showinfo("关于", "LDA 统计模拟演示系统\n课程：统计模拟与数据库编程\n版本：1.0")

    def quit_app(self):
        # 还原标准输出，防止报错
        sys.stdout = self.stdout_backup
        self.destroy()

if __name__ == "__main__":
    app = LDAApp()
    app.mainloop()