import os
import re
import json
import glob
import pandas as pd

# ==========================================
# 1. Configuration & Paths
# ==========================================
difficulty_csv_path = "/project/gcni_compute/qdeng/prompt_eng_hk/revision/code/grading_files/hard_easy_nih.csv"
results_base_dir = "/project/gcni_compute/qdeng/prompt_eng_hk/revision/medgemma27b_nih_results/27b_sort"
output_csv_path = "merged_medgemma27b_nih_results.csv"

model_name = "medgemma27b"
dataset_name = "nih"
pathologies = ["atelectasis", "cardiomegaly", "consolidation", "effusion", "mass", "pneumothorax"]

# ==========================================
# 2. Load Ground Truth and Difficulty CSV
# ==========================================
df_gt = pd.read_csv(difficulty_csv_path)

# Map image_id -> {gt, difficulty}
gt_dict = {}
for _, row in df_gt.iterrows():
    img_id = str(row['id']).strip()
    gt_dict[img_id] = {
        'gt': str(row['gt']).lower().strip(),
        'difficulty': str(row['difficulty']).strip()
    }

# Track all image IDs found in predictions for validation
pred_all_ids = set()

# ==========================================
# 3. Helper: Extract Prediction
# ==========================================
def extract_prediction(response_text):
    """
    Extracts 'Yes', 'No', or 'Uncertainty' from the model response text.
    Handles markdown bolding, punctuation, and fallback edge cases.
    """
    if not isinstance(response_text, str):
        return "Uncertainty"
    
    clean_text = response_text.lower().strip()
    
    # Check for direct word matches
    has_yes = bool(re.search(r'\b(yes)\b', clean_text))
    has_no = bool(re.search(r'\b(no)\b', clean_text))
    has_uncertain = bool(re.search(r'\b(uncertain|uncertainty|unclear|inconclusive|not sure|cannot determine)\b', clean_text))
    
    if has_uncertain:
        return "Uncertainty"
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    
    # If both 'yes' and 'no' are mentioned (e.g. CoT explanations), check the tail of the answer
    if has_yes and has_no:
        last_yes = clean_text.rfind("yes")
        last_no = clean_text.rfind("no")
        return "Yes" if last_yes > last_no else "No"
        
    return "Uncertainty"

# ==========================================
# 4. Parse JSON Predictions
# ==========================================
aggregated_data = {}
json_files = glob.glob(f"{results_base_dir}/**/*.json", recursive=True)

print(f"Found {len(json_files)} JSON files to process...")

for file_path in json_files:
    prompt_type = os.path.basename(os.path.dirname(file_path))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        for item in data:
            # Extract image filename
            image_id = item.get('dir') or os.path.basename(item.get('path', ''))
            image_id = str(image_id).strip()
            # print("item", item)
            if not image_id:
                continue

            pred_all_ids.add(image_id)

            # Determine query disease
            query_disease = item.get('query_disease', '').lower().strip()
            if not query_disease:
                filename = os.path.basename(file_path).lower()
                # print("filename", filename)

                for p in pathologies:
                    if p in filename:
                        query_disease = p
                        # print("query_disease", query_disease)
                        # exit()
                        break

            if not query_disease:
                continue

            pred = extract_prediction(item.get('response', ''))
            # print("pred", pred)
            # exit()
            # Aggregate per (image_id, prompt_type)
            row_key = (image_id, prompt_type)
            if row_key not in aggregated_data:
                gt_info = gt_dict.get(image_id, {'gt': 'unknown', 'difficulty': 'unknown'})
                
                aggregated_data[row_key] = {
                    'id': image_id,
                    'dataset': dataset_name,
                    'gt': gt_info['gt'],
                    'difficulty': gt_info['difficulty'],
                    'model': model_name,
                    'prompt_type': prompt_type
                }
                
                # Initialize default predictions to 'No'
                for p in pathologies:
                    aggregated_data[row_key][f'pred_{p}'] = "No"

            if query_disease in pathologies:
                aggregated_data[row_key][f'pred_{query_disease}'] = pred

# ==========================================
# 5. Build Multi-Hot GT Labels & Finalize Rows
# ==========================================
final_rows = []
for row_data in aggregated_data.values():
    if row_data['difficulty'] == 'unknown':
        continue
    current_gt = row_data['gt']
    
    # Assign multi-hot ground truth (Normal = all 0s)
    for p in pathologies:
        if current_gt == 'normal':
            row_data[f'gt_{p}'] = 0
        elif current_gt == p:
            row_data[f'gt_{p}'] = 1
        else:
            row_data[f'gt_{p}'] = 0
            
    final_rows.append(row_data)

df_final = pd.DataFrame(final_rows)

if not df_final.empty:
    base_cols = ['id', 'dataset', 'gt', 'difficulty', 'model', 'prompt_type']
    gt_cols = [f'gt_{p}' for p in pathologies]
    pred_cols = [f'pred_{p}' for p in pathologies]
    
    df_final = df_final[base_cols + gt_cols + pred_cols]
    df_final.to_csv(output_csv_path, index=False)
    print(f"\n[DONE] Successfully saved {len(df_final)} rows to {output_csv_path}")
else:
    print("\n[WARNING] No rows generated. Please check your results directory path.")

# ==========================================
# 6. Validate Image ID Matching
# ==========================================
print("\n" + "=" * 50)
print("--- Running Image ID Match Check ---")
print("=" * 50)

gt_image_ids = set(gt_dict.keys())

unmatched_in_preds = pred_all_ids - gt_image_ids
unmatched_in_gt = gt_image_ids - pred_all_ids

if unmatched_in_preds:
    print(f"\n[!] Found {len(unmatched_in_preds)} images in prediction files that are MISSING from the grade CSV:")
    for img in sorted(unmatched_in_preds):
        print(f"  - {img}")
else:
    print("\n[✓] All prediction images exist in the grade CSV.")

if unmatched_in_gt:
    print(f"\n[!] Found {len(unmatched_in_gt)} images in the grade CSV that have NO predictions in the JSONs:")
    for img in sorted(unmatched_in_gt):
        print(f"  - {img}")
else:
    print("\n[✓] All grade CSV images have corresponding predictions.")

if not unmatched_in_preds and not unmatched_in_gt:
    print("\n[SUCCESS] Image IDs match completely between prediction JSONs and the grade CSV!")

print("=" * 50)