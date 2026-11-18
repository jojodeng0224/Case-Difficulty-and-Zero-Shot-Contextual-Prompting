import os
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')  # 忽略当某一类样本全为同一分数时的除零警告

# ==============================================================================
# 1. 设置文件路径与输出 TXT 路径
# ==============================================================================
file_path = ""

# 默认保存在同级目录下，命名为 kappa_report.txt
output_txt_path = os.path.join(os.path.dirname(file_path), "kappa_report.txt")

# ==============================================================================
# 2. 定义终端与文件同步输出类 (Logger)
# ==============================================================================
class DualLogger:
    """将所有 print 内容同时输出到终端和指定的 txt 文件中"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# 启动重定向：此后所有的 print() 都会自动写入 txt 文件
logger = DualLogger(output_txt_path)
sys.stdout = logger

# ==============================================================================
# 3. 辅助计算函数
# ==============================================================================
def interpret_kappa(k):
    if pd.isna(k): return "N/A (no variance)"
    if k < 0: return "poor"
    elif k <= 0.20: return "slight"
    elif k <= 0.40: return "fair"
    elif k <= 0.60: return "moderate"
    elif k <= 0.80: return "substantial"
    else: return "almost perfect"

def calc_kappas(group):
    """计算一组数据的4种 Kappa 指标"""
    g_kc = group['grade_kc'].astype(int)
    g_kk = group['grade_kk'].astype(int)
    g_kc_bin = g_kc.apply(lambda x: 'hard' if x <= 3.5 else 'easy')
    g_kk_bin = g_kk.apply(lambda x: 'hard' if x <= 3.5 else 'easy')
    
    q_k = cohen_kappa_score(g_kc, g_kk, weights='quadratic')
    l_k = cohen_kappa_score(g_kc, g_kk, weights='linear')
    u_k = cohen_kappa_score(g_kc, g_kk, weights=None)
    b_k = cohen_kappa_score(g_kc_bin, g_kk_bin)
    return q_k, l_k, u_k, b_k

# ==============================================================================
# 4. 读取数据并输出全部结果
# ==============================================================================
try:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[错误] 文件未找到: {file_path}")

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['grade_kc', 'grade_kk']).copy()

    q_all, l_all, u_all, b_all = calc_kappas(df)

    print("=" * 80)
    print(f"1. 总体 COHEN'S KAPPA 结果 (Total N = {len(df)})  <-- 论文主文本汇报这个")
    print("=" * 80)
    print(f"  * [5-point Scale] Quadratic Weighted κ : {q_all:.4f} ({interpret_kappa(q_all)})")
    print(f"  * [5-point Scale] Linear Weighted κ    : {l_all:.4f} ({interpret_kappa(l_all)})")
    print(f"  * [5-point Scale] Unweighted κ         : {u_all:.4f} ({interpret_kappa(u_all)})")
    print(f"  * [Binary Split]  Easy/Hard κ (<=3.5)  : {b_all:.4f} ({interpret_kappa(b_all)})")

    # 按 Dataset 拆分 (如 nih, padchest)
    if 'dataset' in df.columns:
        print("\n" + "=" * 80)
        print("2. 按 Dataset 拆分结果:")
        print("-" * 80)
        print(f"  {'Dataset':<15} | {'N':<5} | {'Quad κ':<8} | {'Lin κ':<8} | {'Unw κ':<8} | {'Binary κ':<8}")
        print("-" * 80)
        for ds_name, group in df.groupby('dataset'):
            q_k, l_k, u_k, b_k = calc_kappas(group)
            print(f"  {str(ds_name):<15} | {len(group):<5} | {q_k:<8.4f} | {l_k:<8.4f} | {u_k:<8.4f} | {b_k:<8.4f}")

    # 按 Pathology (label 列) 拆分
    if 'label' in df.columns:
        print("\n" + "=" * 80)
        print("3. 按 Pathology (label) 拆分结果:")
        print("-" * 80)
        print(f"  {'Pathology (label)':<20} | {'N':<5} | {'Quad κ':<8} | {'Lin κ':<8} | {'Unw κ':<8} | {'Binary κ':<8}")
        print("-" * 80)
        for label_name, group in df.groupby('label'):
            q_k, l_k, u_k, b_k = calc_kappas(group)
            print(f"  {str(label_name):<20} | {len(group):<5} | {q_k:<8.4f} | {l_k:<8.4f} | {u_k:<8.4f} | {b_k:<8.4f}")

    # 按 Dataset + Pathology 组合拆分
    if 'dataset' in df.columns and 'label' in df.columns:
        print("\n" + "=" * 80)
        print("4. 按 Dataset + Pathology 细分结果:")
        print("-" * 80)
        print(f"  {'Dataset':<12} | {'Pathology':<18} | {'N':<5} | {'Quad κ':<8} | {'Lin κ':<8} | {'Unw κ':<8} | {'Binary κ':<8}")
        print("-" * 80)
        for (ds_name, label_name), group in df.groupby(['dataset', 'label']):
            q_k, l_k, u_k, b_k = calc_kappas(group)
            print(f"  {str(ds_name):<12} | {str(label_name):<18} | {len(group):<5} | {q_k:<8.4f} | {l_k:<8.4f} | {u_k:<8.4f} | {b_k:<8.4f}")

    print("=" * 80)
    print(f"\n[保存成功] 以上所有统计结果已保存至: {output_txt_path}")

finally:
    # 恢复标准输出并关闭文件句柄
    sys.stdout = logger.terminal
    logger.close()
