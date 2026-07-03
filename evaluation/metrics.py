import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Cố định đúng 4 nhãn của ALQAC phục vụ dựng ma trận nhầm lẫn
ALQAC_LABELS = ["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]

def evaluate_alqac_system(gold_data, pred_data):
    """Đánh giá toàn diện hiệu năng hệ thống Agentic RAG theo cấu trúc điểm ALQAC 2026.

    Hàm này thực hiện tính toán điểm số cục bộ (Local Score) mô phỏng chính xác
    100% theo trọng số công thức của Ban tổ chức (70% Outcome Accuracy, 20%
    Penalized Case Recall, và 10% Micro Law Evidence F1). Đồng thời xuất ra 
    ma trận nhầm lẫn để phục vụ chẩn đoán lỗi phân loại nhãn.

    Args:
        gold_data (list of dict): Tập dữ liệu nhãn chuẩn (Ground Truth) do BTC 
            hoặc đội QA tự dán nhãn thủ công. Mỗi phần tử trong danh sách bắt 
            buộc phải có cấu trúc sau:
            {
                "case_id": "0001",
                "prediction": "A_WIN",
                "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}, ...],
                "case_evidence": ["seg1", "seg3", "seg5"],
                "n_segments": 20
            }
        pred_data (list of dict): Tập dữ liệu đầu ra do Pipeline của hệ thống 
            trích xuất và suy luận ra. Mỗi phần tử trong danh sách phải tuân thủ
            cấu trúc sau:
            {
                "case_id": "0001",
                "prediction": "PARTIAL_A_WIN",
                "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}, ...],
                "case_evidence": ["seg1", "seg2"],
                "api_calls": 25
            }

    Returns:
        tuple: Bộ kết quả trả về gồm 2 thành phần:
            - report (dict): Từ điển chứa điểm số Final Score tổng quy đổi và 
              chi tiết từng điểm thành phần cùng các thống kê trích xuất liên quan.
            - cm_df (pd.DataFrame): Ma trận nhầm lẫn (Confusion Matrix) được định 
              dạng dưới dạng Pandas DataFrame với các trục được cố định theo đúng 
              thứ tự 4 nhãn của cuộc thi.

    Raises:
        KeyError: Cảnh báo ra màn hình terminal nếu phát hiện `case_id` có trong 
            tập dữ liệu gốc nhưng bị bỏ sót trong tệp kết quả dự đoán.
    """
    gold_dict = {str(item['case_id']): item for item in gold_data}
    pred_dict = {str(item['case_id']): item for item in pred_data}
    
    total_cases = len(gold_dict)
    correct_outcomes = 0
    total_penalized_case_recall = 0.0
    
    # Tập hợp toàn cục chứa chứng cứ luật phục vụ Micro F1
    G_law_set = set()
    P_law_set = set()
    
    y_true_list = []
    y_pred_list = []
    
    for case_id, gold in gold_dict.items():
        if case_id not in pred_dict:
            print(f"❌ THIẾU CASE_ID: {case_id} trong file dự đoán!")
            continue
            
        pred = pred_dict[case_id]
        
        # Đồng bộ nhãn
        gt_label = str(gold.get('prediction', '')).strip()
        pred_label = str(pred.get('prediction', '')).strip()
        y_true_list.append(gt_label)
        y_pred_list.append(pred_label)
        
        # 1. OUTCOME ACCURACY (Hình image_05fc32.png)
        if gt_label == pred_label:
            correct_outcomes += 1
            
        # 2. PENALIZED CASE EVIDENCE RECALL (Hình image_05fcae.png & image_05fcce.png)
        g_case = set(gold.get('case_evidence', []))
        p_case = set(pred.get('case_evidence', []))
        
        # Tính toán Recall nền tảng (Xử lý an toàn tập dữ liệu rỗng)
        if len(g_case) == 0:
            case_recall = 1.0 if len(p_case) == 0 else 0.0
        else:
            case_recall = len(p_case.intersection(g_case)) / len(g_case)
        
        # Áp dụng công thức max thu gọn chống lỗi ZeroDivision: E_i = max(0, 1 - max(0, c_i - 2n_i) / 3n_i)
        n_i = gold.get('n_segments', 0)
        c_i = pred.get('api_calls', 0)
        
        if n_i == 0:
            E_i = 1.0 if c_i == 0 else 0.0
        else:
            penalty_numerator = max(0, c_i - 2 * n_i)
            penalty_denominator = 3 * n_i
            E_i = max(0.0, 1.0 - (penalty_numerator / penalty_denominator))
            
        total_penalized_case_recall += (case_recall * E_i)
        
        # 3. THU THẬP BẰNG CHỨNG LUẬT CHO MICRO F1 (Hình image_05fceb.png)
        def extract_law_tuples(evidence_list):
            return {f"{case_id}_{e.get('law_id')}_{e.get('aid')}" for e in evidence_list if isinstance(e, dict)}
            
        G_law_set.update(extract_law_tuples(gold.get('law_evidence', [])))
        P_law_set.update(extract_law_tuples(pred.get('law_evidence', [])))

    # --- TÍNH TOÁN CÁC CHỈ SỐ TOÀN CỤC ---
    accuracy = correct_outcomes / total_cases if total_cases > 0 else 0.0
    avg_penalized_case_recall = total_penalized_case_recall / total_cases if total_cases > 0 else 0.0
    
    # Tính toán Micro-averaged F1 chuẩn xác (Hình image_05fceb.png)
    tp_law = len(P_law_set.intersection(G_law_set))
    micro_precision = tp_law / len(P_law_set) if len(P_law_set) > 0 else 0.0
    micro_recall = tp_law / len(G_law_set) if len(G_law_set) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # THÀNH PHẦN ĐIỂM SỐ CUỐI CÙNG: 0.70 * Acc + 0.20 * CaseRecall + 0.10 * LawF1 (Hình image_05fceb.png)
    final_score = (0.70 * accuracy) + (0.20 * avg_penalized_case_recall) + (0.10 * micro_f1)
    
    # Tạo dựng Ma trận nhầm lẫn phục vụ kiểm soát chất lượng
    cm = confusion_matrix(y_true_list, y_pred_list, labels=ALQAC_LABELS)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in ALQAC_LABELS], columns=[f"Pred_{l}" for l in ALQAC_LABELS])
    
    report = {
        "ALQAC_Final_Score": round(final_score, 4),
        "Components": {
            "Outcome_Accuracy_70%": round(accuracy, 4),
            "Penalized_Case_Recall_20%": round(avg_penalized_case_recall, 4),
            "Micro_Law_F1_10%": round(micro_f1, 4)
        },
        "Retrieval_Stats": {
            "Law_Precision": round(micro_precision, 4),
            "Law_Recall": round(micro_recall, 4)
        }
    }
    
    return report, cm_df