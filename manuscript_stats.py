import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import fisher_exact
from scipy.stats import chi2

# ==========================================
# 1. 数据加载与清洗 (隔离 Uncertainty)
# ==========================================
print("Loading data...")
df = pd.read_csv("")

# --- 【功能开关：响应 Reviewer 要求，过滤低 Average 样本】 ---
ENABLE_AVERAGE_FILTER = False  # 设置为 True 开启过滤，False 则保留全集
AVERAGE_COL_NAME = 'average'   # 【重要】请确认您 CSV 中真实的列名 (例如 'average', 'average_score' 等)
AVERAGE_THRESHOLD = 2          # 剔除 <= 2 的样本，即只保留 > 2 的样本

if ENABLE_AVERAGE_FILTER:
    if AVERAGE_COL_NAME in df.columns:
        original_len = len(df)
        df = df[df[AVERAGE_COL_NAME] > AVERAGE_THRESHOLD].copy()
        print(f"\n【Reviewer 要求落实】已剔除 {AVERAGE_COL_NAME} <= {AVERAGE_THRESHOLD} 的样本。")
        print(f"数据量从 {original_len} 减少至 {len(df)}。\n")
    else:
        print(f"\n【警告】未在数据集中找到名为 '{AVERAGE_COL_NAME}' 的列，过滤未生效！请检查变量 AVERAGE_COL_NAME。\n")
# --------------------------------------------------------------

# 定义疾病列表 (根据您的CSV列名提取)
pathologies = ['atelectasis', 'cardiomegaly', 'consolidation', 'effusion', 'mass', 'pneumothorax']

def map_predictions(val):
    val_str = str(val).strip().lower()
    if val_str == 'yes':
        return 1
    elif val_str == 'no':
        return 0
    else:
        # 将 Uncertainty 或其他非预期输出设为缺失值 NaN
        return np.nan 

# 自动清洗所有预测列
for patho in pathologies:
    pred_col = f'pred_{patho}'
    clean_col = f'clean_pred_{patho}'
    if pred_col in df.columns:
        df[clean_col] = df[pred_col].apply(map_predictions)

# ==========================================
# 2. 核心指标计算函数 (仅基于有效数据)
# ==========================================
def calculate_metrics(y_true, y_pred):
    valid_mask = ~np.isnan(y_pred)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    if len(y_true_valid) == 0:
        return np.nan, np.nan, np.nan
        
    tp = np.sum((y_pred_valid == 1) & (y_true_valid == 1))
    tn = np.sum((y_pred_valid == 0) & (y_true_valid == 0))
    fp = np.sum((y_pred_valid == 1) & (y_true_valid == 0))
    fn = np.sum((y_pred_valid == 0) & (y_true_valid == 1))
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    j_index = sens + spec - 1 if (not np.isnan(sens) and not np.isnan(spec)) else np.nan
    
    return sens, spec, j_index

# ==========================================
# 3. 自动化生成点估计与 Bootstrap 置信区间
# ==========================================
print("Calculating Metrics and Bootstrapping CIs...")
n_iterations = 1000
results_list = []

models = df['model'].unique()
difficulties = list(df['difficulty'].unique()) + ['all']
prompts = df['prompt_type'].unique() 

# 统一识别 Baseline Prompt 与 Contextual Prompts (全局复用)
BASE_PROMPT = 'unprompts' if 'unprompts' in prompts else ('unprompted' if 'unprompted' in prompts else prompts[0])
contextual_prompts = [p for p in prompts if p != BASE_PROMPT]

np.random.seed(42)
for patho in pathologies:
    gt_col = f'gt_{patho}'
    pred_col = f'clean_pred_{patho}'
    
    for model in models:
        for diff in difficulties:
            for prompt in prompts:
                if diff == 'all':
                    subset = df[(df['model'] == model) & (df['prompt_type'] == prompt)]
                else:
                    subset = df[(df['model'] == model) & (df['difficulty'] == diff) & (df['prompt_type'] == prompt)]
                
                if patho == 'atelectasis' and model == 'medgemma4b' and prompt == 'unprompts':
                    normal_in_subset = subset[
                        (subset['gt_atelectasis'] == 0) & (subset['gt_cardiomegaly'] == 0) & 
                        (subset['gt_consolidation'] == 0) & (subset['gt_effusion'] == 0) & 
                        (subset['gt_mass'] == 0) & (subset['gt_pneumothorax'] == 0)
                    ]
                    print(f"【验证结果】在 {diff} 难度的 {model} {prompt} 子集中，共包含了 {len(normal_in_subset)} 张正常图像。")
                
                if subset.empty:
                    continue
                
                y_true = subset[gt_col].values
                y_pred = subset[pred_col].values
                
                sens, spec, j_idx = calculate_metrics(y_true, y_pred)
                
                bootstrapped_sens, bootstrapped_spec, bootstrapped_j = [], [], []
                
                pos_indices = np.where(y_true == 1)[0]
                neg_indices = np.where(y_true == 0)[0]
                n_pos = len(pos_indices)
                n_neg = len(neg_indices)
                
                for _ in range(n_iterations):
                    if n_pos > 0:
                        boot_pos = np.random.choice(pos_indices, size=n_pos, replace=True)
                    else:
                        boot_pos = np.array([], dtype=int)
                        
                    if n_neg > 0:
                        boot_neg = np.random.choice(neg_indices, size=n_neg, replace=True)
                    else:
                        boot_neg = np.array([], dtype=int)
                    
                    boot_indices = np.concatenate([boot_pos, boot_neg])
                    y_t_boot = y_true[boot_indices]
                    y_p_boot = y_pred[boot_indices]
                    
                    b_sens, b_spec, b_j = calculate_metrics(y_t_boot, y_p_boot)
                    if not np.isnan(b_j):
                        bootstrapped_sens.append(b_sens)
                        bootstrapped_spec.append(b_spec)
                        bootstrapped_j.append(b_j)
                
                ci_sens = f"[{np.percentile(bootstrapped_sens, 2.5):.3f}, {np.percentile(bootstrapped_sens, 97.5):.3f}]" if bootstrapped_sens else "[NaN, NaN]"
                ci_spec = f"[{np.percentile(bootstrapped_spec, 2.5):.3f}, {np.percentile(bootstrapped_spec, 97.5):.3f}]" if bootstrapped_spec else "[NaN, NaN]"
                ci_j = f"[{np.percentile(bootstrapped_j, 2.5):.3f}, {np.percentile(bootstrapped_j, 97.5):.3f}]" if bootstrapped_j else "[NaN, NaN]"
                
                sens_q1 = np.percentile(bootstrapped_sens, 25) if bootstrapped_sens else np.nan
                sens_q3 = np.percentile(bootstrapped_sens, 75) if bootstrapped_sens else np.nan
                
                spec_q1 = np.percentile(bootstrapped_spec, 25) if bootstrapped_spec else np.nan
                spec_q3 = np.percentile(bootstrapped_spec, 75) if bootstrapped_spec else np.nan
                
                j_q1 = np.percentile(bootstrapped_j, 25) if bootstrapped_j else np.nan
                j_q3 = np.percentile(bootstrapped_j, 75) if bootstrapped_j else np.nan

                sens_med = np.percentile(bootstrapped_sens, 50) if bootstrapped_sens else np.nan
                spec_med = np.percentile(bootstrapped_spec, 50) if bootstrapped_spec else np.nan
                j_med = np.percentile(bootstrapped_j, 50) if bootstrapped_j else np.nan

                valid_n = np.sum(~np.isnan(y_pred))

                if diff != 'all':
                    final_spec = "N/A (Confounded)"
                    final_j = "N/A (Confounded)"
                    spec_q1, spec_med, spec_q3 = np.nan, np.nan, np.nan
                    j_q1, j_med, j_q3 = np.nan, np.nan, np.nan
                else:
                    final_spec = f"{spec:.3f} {ci_spec}"
                    final_j = f"{j_idx:.3f} {ci_j}"
                
                results_list.append({
                    'Pathology': patho,
                    'Model': model,
                    'Difficulty': diff,
                    'Prompt': prompt,
                    'Valid_N': valid_n,
                    'Sensitivity': f"{sens:.3f} {ci_sens}",
                    'Sens_Q1': sens_q1,
                    'Sens_Med': sens_med,  
                    'Sens_Q3': sens_q3,
                    'Specificity': final_spec,
                    'Spec_Q1': spec_q1,
                    'Spec_Med': spec_med,  
                    'Spec_Q3': spec_q3,
                    'Youden_J': final_j,
                    'J_Q1': j_q1,
                    'J_Med': j_med,        
                    'J_Q3': j_q3
                })

df_metrics = pd.DataFrame(results_list)
df_metrics.to_csv("metrics_with_bootstrap_ci_q1_q3.csv", index=False)
print("Metrics saved to metrics_with_bootstrap_ci_q1_q3.csv")

# ==========================================
# 4. 自动化进行 McNemar & Paired Bootstrap 检验并做 FDR 校正
#    (按老板要求：仅比较 Contextual Prompt vs. Baseline，不比较 Instruction vs. GK)
# ==========================================
print("Running Pairwise Tests (Contextual Prompts vs. Baseline: McNemar & Paired Bootstrap for Youden's J)...")
mcnemar_results = []

ENABLE_COCHRAN_Q = True  
Q_ALPHA = 0.05  
N_BOOT_PAIRED = 10000  # 10,000 次向量化抽样，确保经验 P 值具有极高分辨率以配合 FDR 排序

cochrans_q_results = []  

def cochrans_q_test(outcomes):
    n, k = outcomes.shape
    if k < 2 or n == 0:
        return np.nan, 1.0, k - 1
        
    row_sums = np.sum(outcomes, axis=1)
    col_sums = np.sum(outcomes, axis=0)
    
    total_sum = np.sum(col_sums)
    sum_col_sums_sq = np.sum(col_sums ** 2)
    sum_row_sums_sq = np.sum(row_sums ** 2)
    
    denominator = k * total_sum - sum_row_sums_sq
    if denominator == 0:
        return 0.0, 1.0, k - 1
        
    q_stat = ((k - 1) * (k * sum_col_sums_sq - total_sum ** 2)) / denominator
    p_val = chi2.sf(q_stat, k - 1)
    return q_stat, p_val, k - 1

def run_mcnemar_comparison(df_base, df_other, comp_name, comp_type, patho, model_name, diff, prompt_name, skip_sens=False, skip_spec=False):
    if df_base.empty or df_other.empty:
        return
    
    merged = pd.merge(df_base[['id', gt_col, pred_col]], 
                      df_other[['id', gt_col, pred_col]], 
                      on='id', suffixes=('_base', '_other'))
    
    valid_mask = ~merged[f'{pred_col}_base'].isna() & ~merged[f'{pred_col}_other'].isna()
    merged_valid = merged[valid_mask]
    
    if len(merged_valid) == 0:
        return
    
    gt_values = merged_valid[f'{gt_col}_base'].values
    pred_base = merged_valid[f'{pred_col}_base'].values
    pred_other = merged_valid[f'{pred_col}_other'].values
    
    # ---------------------------------------------------------
    # 检验 1: Sensitivity (GT == 1) via McNemar's Exact Test
    # ---------------------------------------------------------
    if not skip_sens:
        pos_mask = (gt_values == 1)
        if np.sum(pos_mask) > 0:
            corr_b = (pred_base[pos_mask] == 1)
            corr_o = (pred_other[pos_mask] == 1)
            table = [[np.sum(corr_b & corr_o), np.sum(corr_b & ~corr_o)],
                     [np.sum(~corr_b & corr_o), np.sum(~corr_b & ~corr_o)]]
            try:
                p_sens = mcnemar(table, exact=True).pvalue
            except ValueError:
                p_sens = 1.0
            
            mcnemar_results.append({
                'Pathology': patho, 'Difficulty': diff, 'Comparison_Type': comp_type,
                'Model': model_name, 'Prompt': prompt_name, 'Comparison': comp_name,
                'Metric': 'Sensitivity', 'Valid_Paired_N': np.sum(pos_mask), 'Raw_P_Value': p_sens
            })

    # ---------------------------------------------------------
    # 检验 2: Specificity (GT == 0) via McNemar's Exact Test
    # ---------------------------------------------------------
    if diff == 'all' and not skip_spec:
        neg_mask = (gt_values == 0)
        if np.sum(neg_mask) > 0:
            corr_b = (pred_base[neg_mask] == 0)
            corr_o = (pred_other[neg_mask] == 0)
            table = [[np.sum(corr_b & corr_o), np.sum(corr_b & ~corr_o)],
                     [np.sum(~corr_b & corr_o), np.sum(~corr_b & ~corr_o)]]
            try:
                p_spec = mcnemar(table, exact=True).pvalue
            except ValueError:
                p_spec = 1.0
            
            mcnemar_results.append({
                'Pathology': patho, 'Difficulty': diff, 'Comparison_Type': comp_type,
                'Model': model_name, 'Prompt': prompt_name, 'Comparison': comp_name,
                'Metric': 'Specificity', 'Valid_Paired_N': np.sum(neg_mask), 'Raw_P_Value': p_spec
            })

    # ---------------------------------------------------------
    # 检验 3: Youden's J via Paired Stratified Bootstrap Test
    # 只要 Sensitivity 或 Specificity 至少有一个通过了 Cochran's Q 把关，即可检验 Youden's J
    # ---------------------------------------------------------
    if diff == 'all' and not (skip_sens and skip_spec):
        pos_mask = (gt_values == 1)
        neg_mask = (gt_values == 0)
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)
        
        if n_pos > 0 and n_neg > 0:
            pb_pos = pred_base[pos_mask]
            po_pos = pred_other[pos_mask]
            pb_neg = pred_base[neg_mask]
            po_neg = pred_other[neg_mask]
            
            # 如果两个模型在所有有效样本上的预测完全一致，差值恒为 0，P 值直接记为 1.0
            if np.array_equal(pb_pos, po_pos) and np.array_equal(pb_neg, po_neg):
                p_j = 1.0
            else:
                # 向量化分层配对重抽样：同时为两个模型抽取完全相同的样本索引
                pos_boot_idx = np.random.randint(0, n_pos, size=(N_BOOT_PAIRED, n_pos))
                neg_boot_idx = np.random.randint(0, n_neg, size=(N_BOOT_PAIRED, n_neg))
                
                # 计算 10,000 次抽样下各自的 Sensitivity 和 Specificity
                sens_b_boot = np.mean(pb_pos[pos_boot_idx] == 1, axis=1)
                sens_o_boot = np.mean(po_pos[pos_boot_idx] == 1, axis=1)
                
                spec_b_boot = np.mean(pb_neg[neg_boot_idx] == 0, axis=1)
                spec_o_boot = np.mean(po_neg[neg_boot_idx] == 0, axis=1)
                
                # 计算配对的 Youden's J 差值分布: Delta_J = J_base - J_other
                delta_j = (sens_b_boot + spec_b_boot - 1.0) - (sens_o_boot + spec_o_boot - 1.0)
                
                # 采用 Phipson & Smyth (2010) 修正公式计算双侧经验 P 值，避免出现绝对 0 值
                count_le_zero = np.sum(delta_j <= 0)
                count_ge_zero = np.sum(delta_j >= 0)
                p_j = 2.0 * (min(count_le_zero, count_ge_zero) + 1.0) / (N_BOOT_PAIRED + 1.0)
                p_j = min(p_j, 1.0)
                
            mcnemar_results.append({
                'Pathology': patho, 'Difficulty': diff, 'Comparison_Type': comp_type,
                'Model': model_name, 'Prompt': prompt_name, 'Comparison': comp_name,
                'Metric': 'Youden_J', 'Valid_Paired_N': n_pos + n_neg, 'Raw_P_Value': p_j
            })

# ==================== 主测试循环 ====================
np.random.seed(42)  # 固定随机种子，确保 Paired Bootstrap 的 P 值 100% 可复现

for patho in pathologies:
    gt_col = f'gt_{patho}'
    pred_col = f'clean_pred_{patho}'
    
    for diff in difficulties:
        
        # 对比 A: Prompt_Eval (同一模型，仅做 Contextual Prompt vs. Baseline 的配对比较)
        for model in models:
            skip_sens, skip_spec = False, False
            
            if ENABLE_COCHRAN_Q and len(prompts) >= 3:
                dfs_p = [df[(df['model'] == model) & (df['prompt_type'] == p) & (df['difficulty'] == diff if diff != 'all' else True)] for p in prompts]
                combined = pd.concat([d.set_index('id')[pred_col] for d in dfs_p], axis=1).dropna()
                if not combined.empty:
                    gt_series = dfs_p[0].set_index('id')[gt_col].loc[combined.index]
                    
                    pos_mask = (gt_series == 1).values
                    valid_n_pos = np.sum(pos_mask)
                    if valid_n_pos > 0:
                        outcomes = (combined.values[pos_mask] == 1).astype(int)
                        q_stat, q_p, q_df = cochrans_q_test(outcomes)
                        
                        cochrans_q_results.append({
                            'Pathology': patho, 'Difficulty': diff, 'Comparison_Type': 'Prompt_Eval',
                            'Base_Group': model, 'Metric': 'Sensitivity', 'N_Groups': len(prompts),
                            'Valid_N': valid_n_pos, 'Q_Statistic': q_stat, 'df': q_df, 'P_Value': q_p
                        })
                        if q_p >= Q_ALPHA: skip_sens = True
                            
                    if diff == 'all':
                        neg_mask = (gt_series == 0).values
                        valid_n_neg = np.sum(neg_mask)
                        if valid_n_neg > 0:
                            outcomes = (combined.values[neg_mask] == 0).astype(int)
                            q_stat, q_p, q_df = cochrans_q_test(outcomes)
                            
                            cochrans_q_results.append({
                                'Pathology': patho, 'Difficulty': diff, 'Comparison_Type': 'Prompt_Eval',
                                'Base_Group': model, 'Metric': 'Specificity', 'N_Groups': len(prompts),
                                'Valid_N': valid_n_neg, 'Q_Statistic': q_stat, 'df': q_df, 'P_Value': q_p
                            })
                            if q_p >= Q_ALPHA: skip_spec = True

            # 【修改点 1】：仅遍历 contextual_prompts 与 BASE_PROMPT 对比，不再比较 instruction vs Knowledge_Prompt
            df_base_p = df[(df['model'] == model) & (df['prompt_type'] == BASE_PROMPT)]
            if diff != 'all':
                df_base_p = df_base_p[df_base_p['difficulty'] == diff]

            for ctx_p in contextual_prompts:
                df_ctx_p = df[(df['model'] == model) & (df['prompt_type'] == ctx_p)]
                if diff != 'all':
                    df_ctx_p = df_ctx_p[df_ctx_p['difficulty'] == diff]
                run_mcnemar_comparison(
                    df_base_p, df_ctx_p, f"{BASE_PROMPT} vs {ctx_p}",
                    "Prompt_Eval", patho, model, diff, "Multiple", skip_sens, skip_spec
                )

        # 对比 B: Model_Eval (已保持注释状态)

if cochrans_q_results:
    df_q = pd.DataFrame(cochrans_q_results)
    df_q.to_csv("cochrans_q_tests.csv", index=False)
    print("Cochran's Q tests saved to cochrans_q_tests.csv")

print("Total pairwise tests conducted:", len(mcnemar_results))
df_mcnemar = pd.DataFrame(mcnemar_results)
if not df_mcnemar.empty:
    p_values = df_mcnemar['Raw_P_Value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    df_mcnemar['FDR_Adjusted_P'] = pvals_corrected
    df_mcnemar['Significant'] = reject

df_mcnemar.to_csv("mcnemar_tests_fdr_corrected.csv", index=False)
print("Pairwise tests (including Youden's J) saved to mcnemar_tests_fdr_corrected.csv")

# ==========================================
# 5. 同 Prompt 内双侧 Fisher 精确检验 (n=72) & 跨难度标准非劣效性检验 (Non-Inferiority)
#    (按老板要求：跨难度比较仅进行单侧 Bootstrap 非劣效性检验，不再进行跨层 Fisher 检验)
# ==========================================
print("\nRunning Within-Prompt Fisher's Exact Tests & Cross-Strata Non-Inferiority Tests...")

# --- 【核心参数配置】 ---
# 1. 门控设置：True 表示要求 Hard 组内 McNemar 的 FDR_Adjusted_P < 0.05 才触发；
#    False 表示只要通过 Cochran's Q 且 McNemar Raw_P_Value < 0.05 即触发。
USE_FDR_FOR_MCNEMAR_TRIGGER = True  

# 2. 预设的非劣效性界值 (Margin \delta)：
#    NI_MARGIN = 0.10 表示容忍下限为 -10% (-0.10)。
NI_MARGIN = 0.10  
N_BOOT_NI = 10000  # 10,000 次重抽样计算差值的单侧 95% 置信区间下限与非劣效 P 值
# ------------------------

print(f"Detected Baseline Prompt: '{BASE_PROMPT}' | Contextual Prompts: {contextual_prompts}")
print(f"Non-Inferiority Margin (delta): {NI_MARGIN:.2f} (Allowed lower bound: {-NI_MARGIN:.2f})")

std_fisher_results = []  # 独立存放同 Prompt 内的 Easy vs Hard 双侧 Fisher 检验结果 (n = 72)
cross_ni_results = []    # 独立存放 McNemar 门控触发的跨难度单侧 Bootstrap 非劣效性检验结果
np.random.seed(42)       # 固定随机种子，确保非劣效 Bootstrap 100% 可复现

for patho in pathologies:
    gt_col = f'gt_{patho}'
    pred_col = f'clean_pred_{patho}'
    
    for model in models:
        # -----------------------------------------------------------------
        # 对比 1: 常规同 Prompt 下的 Easy vs Hard 对比 (Two-Sided Fisher)
        # -----------------------------------------------------------------
        for prompt in prompts:
            df_easy = df[(df['model'] == model) & (df['prompt_type'] == prompt) & (df['difficulty'] == 'easy')]
            df_hard = df[(df['model'] == model) & (df['prompt_type'] == prompt) & (df['difficulty'] == 'hard')]
            
            if df_easy.empty or df_hard.empty:
                continue
                
            easy_valid = df_easy[~df_easy[pred_col].isna() & (df_easy[gt_col] == 1)]
            hard_valid = df_hard[~df_hard[pred_col].isna() & (df_hard[gt_col] == 1)]
            
            if len(easy_valid) == 0 or len(hard_valid) == 0:
                continue
                
            easy_correct = np.sum(easy_valid[pred_col] == 1)
            easy_wrong = np.sum(easy_valid[pred_col] == 0)
            
            hard_correct = np.sum(hard_valid[pred_col] == 1)
            hard_wrong = np.sum(hard_valid[pred_col] == 0)
            
            sens_easy = easy_correct / len(easy_valid)
            sens_hard = hard_correct / len(hard_valid)
            
            table = [[easy_correct, hard_correct],
                     [easy_wrong, hard_wrong]]
            
            try:
                res = fisher_exact(table, alternative='two-sided')
                p_val = res[1]
            except Exception:
                p_val = 1.0
                
            std_fisher_results.append({
                'Pathology': patho,
                'Model': model,
                'Comparison_Type': 'Standard_Easy_vs_Hard',
                'Prompt': prompt,
                'Comparison': f'Easy ({prompt}) vs Hard ({prompt})',
                'Metric': 'Sensitivity',
                'Easy_N': len(easy_valid),
                'Hard_N': len(hard_valid),
                'Easy_Sens': round(sens_easy, 4),
                'Hard_Sens': round(sens_hard, 4),
                'Sens_Diff (Hard - Easy)': round(sens_hard - sens_easy, 4),
                'Raw_P_Value': p_val
            })

        # -----------------------------------------------------------------
        # 对比 2: 基于 McNemar 显著提升门控的跨难度标准非劣效性检验 (One-Sided Bootstrap)
        # 【修改点 2】：不再计算跨层双侧 Fisher，仅计算单侧 Bootstrap 非劣效性检验！
        # -----------------------------------------------------------------
        df_hard_base = df[(df['model'] == model) & (df['prompt_type'] == BASE_PROMPT) & (df['difficulty'] == 'hard')]
        df_easy_base = df[(df['model'] == model) & (df['prompt_type'] == BASE_PROMPT) & (df['difficulty'] == 'easy')]
        
        hard_base_valid = df_hard_base[~df_hard_base[pred_col].isna() & (df_hard_base[gt_col] == 1)]
        easy_base_valid = df_easy_base[~df_easy_base[pred_col].isna() & (df_easy_base[gt_col] == 1)]
        
        if len(hard_base_valid) > 0 and len(easy_base_valid) > 0 and not df_mcnemar.empty:
            sens_hard_base = np.mean(hard_base_valid[pred_col] == 1)
            
            easy_base_preds = easy_base_valid[pred_col].values
            n_easy = len(easy_base_preds)
            sens_easy_base = np.mean(easy_base_preds == 1)
            
            for ctx_prompt in contextual_prompts:
                df_hard_ctx = df[(df['model'] == model) & (df['prompt_type'] == ctx_prompt) & (df['difficulty'] == 'hard')]
                hard_ctx_valid = df_hard_ctx[~df_hard_ctx[pred_col].isna() & (df_hard_ctx[gt_col] == 1)]
                
                if len(hard_ctx_valid) == 0:
                    continue
                
                hard_ctx_preds = hard_ctx_valid[pred_col].values
                n_hard = len(hard_ctx_preds)
                sens_hard_ctx = np.mean(hard_ctx_preds == 1)
                
                # 去 df_mcnemar 中查找该模型在 hard 组下 BASE_PROMPT 与 ctx_prompt 的配对检验结果
                possible_comp_names = [f"{BASE_PROMPT} vs {ctx_prompt}", f"{ctx_prompt} vs {BASE_PROMPT}"]
                mcnemar_match = df_mcnemar[
                    (df_mcnemar['Pathology'] == patho) &
                    (df_mcnemar['Difficulty'] == 'hard') &
                    (df_mcnemar['Comparison_Type'] == 'Prompt_Eval') &
                    (df_mcnemar['Model'] == model) &
                    (df_mcnemar['Metric'] == 'Sensitivity') &
                    (df_mcnemar['Comparison'].isin(possible_comp_names))
                ]
                
                if not mcnemar_match.empty:
                    mc_row = mcnemar_match.iloc[0]
                    mc_raw_p = mc_row['Raw_P_Value']
                    mc_fdr_p = mc_row['FDR_Adjusted_P']
                    mc_is_sig = mc_row['Significant'] if USE_FDR_FOR_MCNEMAR_TRIGGER else (mc_raw_p < 0.05)
                    
                    # 【核心双门控触发】：不仅要敏感度高于 Hard Baseline，还必须通过 McNemar 显著性检验！
                    if (sens_hard_ctx > sens_hard_base) and mc_is_sig:
                        # 向量化 Bootstrap (10,000次) 计算两组差值 Sens_Hard_Ctx - Sens_Easy_Base 及其单侧 95% 置信区间
                        boot_hard_idx = np.random.randint(0, n_hard, size=(N_BOOT_NI, n_hard))
                        boot_easy_idx = np.random.randint(0, n_easy, size=(N_BOOT_NI, n_easy))
                        
                        boot_sens_hard = np.mean(hard_ctx_preds[boot_hard_idx] == 1, axis=1)
                        boot_sens_easy = np.mean(easy_base_preds[boot_easy_idx] == 1, axis=1)
                        boot_diff = boot_sens_hard - boot_sens_easy
                        
                        # 计算 5% 分位数（即单侧 95% 置信区间下限）和 95% 分位数（构成双侧 90% CI）
                        ci_lower_95_onesided = np.percentile(boot_diff, 5.0)
                        ci_upper_90_twosided = np.percentile(boot_diff, 95.0)
                        ci_90_str = f"[{ci_lower_95_onesided:.3f}, {ci_upper_90_twosided:.3f}]"
                        
                        # 与预设非劣效界值 (-NI_MARGIN) 比较
                        ni_pass_ci = bool(ci_lower_95_onesided > -NI_MARGIN)
                        
                        # 计算对应的单侧非劣效 P 值 (H0: Sens_Hard - Sens_Easy <= -NI_MARGIN)，含 Phipson-Smyth 修正
                        count_inferior = np.sum(boot_diff <= -NI_MARGIN)
                        p_ni_raw = (count_inferior + 1.0) / (N_BOOT_NI + 1.0)
                        
                        cross_ni_results.append({
                            'Pathology': patho,
                            'Model': model,
                            'Comparison_Type': 'Cross_HardCtx_vs_EasyBase',
                            'Prompt': f'{ctx_prompt} vs {BASE_PROMPT}',
                            'Comparison': f'Hard ({ctx_prompt}) vs Easy ({BASE_PROMPT})',
                            'Metric': 'Sensitivity',
                            'Easy_N': n_easy,
                            'Hard_N': n_hard,
                            'Easy_Sens': round(sens_easy_base, 4),
                            'Hard_Sens': round(sens_hard_ctx, 4),
                            'Hard_Base_Sens': round(sens_hard_base, 4),
                            'Hard_McNemar_Raw_P': round(mc_raw_p, 6),
                            'Hard_McNemar_FDR_P': round(mc_fdr_p, 6),
                            'Sens_Diff (Hard - Easy)': round(sens_hard_ctx - sens_easy_base, 4),
                            # --- 标准非劣效性检验输出列 ---
                            'NI_Margin_Delta': NI_MARGIN,
                            'NI_95%_OneSided_Lower_Bound': round(ci_lower_95_onesided, 4),
                            'NI_TwoSided_90%_CI': ci_90_str,
                            'NI_Pass_By_CI_Lower_Bound': ni_pass_ci,
                            'NI_Raw_P_Value': round(p_ni_raw, 6)
                        })

# --- 1. 独立对常规同 Prompt 内的 Easy vs Hard (n = 72) 进行双侧 Fisher 的 BH-FDR 校正 ---
df_std = pd.DataFrame(std_fisher_results)
if not df_std.empty:
    std_p_vals = df_std['Raw_P_Value'].values.astype(float)
    rej_std, pvals_std_fdr, _, _ = multipletests(std_p_vals, alpha=0.05, method='fdr_bh')
    df_std['FDR_Adjusted_P'] = pvals_std_fdr
    df_std['Significant'] = rej_std
    df_std.to_csv("fisher_tests_easy_vs_hard.csv", index=False)

# --- 2. 独立对触发的跨难度对比进行单侧 Bootstrap 非劣效性 P 值的 BH-FDR 校正 ---
df_cross = pd.DataFrame(cross_ni_results)
if not df_cross.empty:
    ni_raw_p_vals = df_cross['NI_Raw_P_Value'].values.astype(float)
    rej_ni, p_ni_fdr, _, _ = multipletests(ni_raw_p_vals, alpha=0.05, method='fdr_bh')
    df_cross['NI_FDR_Adjusted_P'] = np.round(p_ni_fdr, 6)
    df_cross['NI_Significant_FDR'] = rej_ni
    
    # 按照论文正文的双重严格标准生成结论列 (单侧 95% CI 下限 > -10% 且 FDR 校正后 P < 0.05)
    conclusions = []
    for _, row in df_cross.iterrows():
        ni_ci_ok = bool(row['NI_Pass_By_CI_Lower_Bound'])
        ni_fdr_ok = bool(row['NI_Significant_FDR'])
        lb = row['NI_95%_OneSided_Lower_Bound']
        
        if ni_ci_ok and ni_fdr_ok:
            conclusions.append(
                f"Non-Inferior Established (证明非劣效: 单侧95%CI下限 {lb:.1%} > 界值 {-NI_MARGIN:.0%}, 且 NI_P_FDR < 0.05)"
            )
        elif ni_ci_ok and not ni_fdr_ok:
            conclusions.append(
                f"Borderline / Unadjusted Pass Only (单侧95%CI下限 {lb:.1%} > {-NI_MARGIN:.0%}, 但 FDR 校正后 P={row['NI_FDR_Adjusted_P']:.4f} >= 0.05)"
            )
        else:
            conclusions.append(
                f"Non-Inferiority Not Established (未达非劣效标准: 单侧95%CI下限 {lb:.1%} <= {-NI_MARGIN:.0%})"
            )
            
    df_cross['Conclusion_Summary'] = conclusions

df_cross.to_csv("fisher_tests_hard_contextual_vs_easy_baseline.csv", index=False)

# 同步保存一份合并归档表 (仅供归档留存，两个家族的 FDR 已各自严格独立完成)
df_fisher_all = pd.concat([df_std, df_cross], ignore_index=True)
df_fisher_all.to_csv("fisher_tests_all_global_fdr.csv", index=False)

print(f"Within-prompt Fisher tests (Easy vs Hard) FDR-adjusted: {len(df_std)}")
print(f"McNemar-Gated Cross-difficulty Non-Inferiority tests FDR-adjusted: {len(df_cross)}")
print("Saved to 'fisher_tests_easy_vs_hard.csv', 'fisher_tests_hard_contextual_vs_easy_baseline.csv', and 'fisher_tests_all_global_fdr.csv'.")

# =========================================================================
# 6. 打印论文 Methods 所需的精确检验次数明细 (Exact Test Counts for Manuscript)
# =========================================================================
print("\n" + "="*70)
print("EXACT NUMBER OF TESTS FOR MANUSCRIPT REPORTING (3 DISTINCT FAMILIES)")
print("="*70)

# 检验簇 (1): 非配对同 Prompt 内 Easy vs. Hard 双侧 Fisher 精确检验簇
n_fisher_std = len(df_std)
print(f"[Family 1 - Unpaired Within-Prompt Fisher Family] Total pooled for FDR: n = {n_fisher_std}")

# 检验簇 (2): 配对事后比较检验簇 (仅 Contextual Prompt vs. Baseline)
n_paired_total = len(df_mcnemar)
if not df_mcnemar.empty:
    n_paired_sens = len(df_mcnemar[df_mcnemar['Metric'] == 'Sensitivity'])
    n_paired_spec = len(df_mcnemar[df_mcnemar['Metric'] == 'Specificity'])
    n_paired_j = len(df_mcnemar[df_mcnemar['Metric'] == 'Youden_J'])
    
    n_sens_easy = len(df_mcnemar[(df_mcnemar['Metric'] == 'Sensitivity') & (df_mcnemar['Difficulty'] == 'easy')])
    n_sens_hard = len(df_mcnemar[(df_mcnemar['Metric'] == 'Sensitivity') & (df_mcnemar['Difficulty'] == 'hard')])
    n_sens_all  = len(df_mcnemar[(df_mcnemar['Metric'] == 'Sensitivity') & (df_mcnemar['Difficulty'] == 'all')])
    
    print(f"\n[Family 2 - Paired Prompt-vs-Baseline Family] Total pooled for FDR (after Cochran's Q gating): n = {n_paired_total}")
    print(f"   ├── Exact McNemar (Sensitivity): n = {n_paired_sens} (Easy: {n_sens_easy}, Hard: {n_sens_hard}, All: {n_sens_all})")
    print(f"   ├── Exact McNemar (Specificity, All): n = {n_paired_spec}")
    print(f"   └── Paired Bootstrap (Youden's J, All): n = {n_paired_j}")

# 检验簇 (3): 跨层单侧 Bootstrap 非劣效性检验簇
n_ni_cross = len(df_cross)
n_ni_sig = int(df_cross['NI_Significant_FDR'].sum()) if (not df_cross.empty and 'NI_Significant_FDR' in df_cross.columns) else 0
print(f"\n[Family 3 - Cross-Strata Non-Inferiority Bootstrap Family] Total pooled for FDR: n = {n_ni_cross} (Established Non-Inferior: {n_ni_sig})")

if cochrans_q_results:
    print(f"\n[Omnibus Gatekeeper] Cochran's Q tests conducted (unadjusted): n = {len(df_q)}")
print("="*70 + "\n")
