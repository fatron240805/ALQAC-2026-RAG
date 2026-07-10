from __future__ import annotations

from typing import Any

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    from sklearn.metrics import confusion_matrix  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    confusion_matrix = None

# Cố định đúng 4 nhãn chuẩn của ALQAC theo đặc tả của BTC
ALQAC_LABELS = ["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]


class SimpleConfusionMatrix:
    """Small printable fallback when pandas/sklearn are not installed."""

    def __init__(self, labels: list[str], matrix: list[list[int]]) -> None:
        self.labels = labels
        self.matrix = matrix

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            f"True_{true_label}": {
                f"Pred_{pred_label}": self.matrix[row][col]
                for col, pred_label in enumerate(self.labels)
            }
            for row, true_label in enumerate(self.labels)
        }

    def __str__(self) -> str:
        return str(self.to_dict())


def _build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> Any:
    if confusion_matrix is not None and pd is not None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return pd.DataFrame(
            cm,
            index=[f"True_{label}" for label in labels],
            columns=[f"Pred_{label}" for label in labels],
        )

    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in label_to_index and pred_label in label_to_index:
            matrix[label_to_index[true_label]][label_to_index[pred_label]] += 1
    return SimpleConfusionMatrix(labels, matrix)

def evaluate_alqac_system(gold_data, pred_data):
    """Bộ đánh giá Local hoàn chỉnh mô phỏng chính xác 100% hệ thống ALQAC 2026.

    Hàm này thực hiện tính toán điểm số tổng hợp (Final Score) nội bộ dựa trên 3 
    cấu phần cốt lõi của cuộc thi với tỷ lệ trọng số 70-20-10. Nó xử lý triệt để 
    các điều kiện biên đặc biệt (Edge Cases) nhằm đảm bảo la bàn local khớp từng 
    milimét với hệ thống chấm điểm thực tế trên máy chủ Leaderboard.

    Cấu phần chỉ số chi tiết:
        1. Outcome Accuracy (Trọng số 70%): Đo lường chính xác tỷ lệ phân loại nhãn 
           phán quyết (Exact Match) qua 4 nhãn tiêu chuẩn: A_WIN (Nguyên đơn thắng), 
           PARTIAL_A_WIN (Nguyên đơn thắng một phần), B_WIN (Bị đơn thắng), và 
           PARTIAL_B_WIN (Bị đơn thắng một phần).
        2. Penalized Case Evidence Recall (Trọng số 20%): Tỷ lệ trích xuất phân đoạn 
           tình tiết vụ án (Case Evidence Recall) nhân với Hệ số phạt hiệu năng API 
           (Penalty Factor E_i). Theo quy chế BTC, số lượng bằng chứng trùng lặp 
           (duplicate ids) sẽ bị tự động loại bỏ (De-duplicated) trước khi chấm.
        3. Micro Law Evidence F1 (Trọng số 10%): Đo lường năng lực tìm kiếm văn bản 
           luật pháp dựa trên phép toán Hợp toàn cục (Global Union) trên toàn bộ tập 
           kiểm thử, loại bỏ sự phụ thuộc vào mã vụ án (case_id).

    Args:
        gold_data (list of dict): Tập dữ liệu nhãn chuẩn (Ground Truth nội bộ) do 
            đội ngũ QA dán nhãn hoặc tập Public mẫu. Mỗi dict phần tử bắt buộc 
            phải chứa cấu trúc:
            {
                "case_id": "case_4101",
                "prediction": "PARTIAL_A_WIN",
                "case_evidence": ["case_4101_chunk_3", "case_4101_chunk_5"],
                "law_evidence": [{"law_id": "91/2015/QH13", "aid": 15}, ...]
            }
        pred_data (list of dict): Tập dữ liệu log chạy đầy đủ do Pipeline Agentic RAG 
            của hệ thống trích xuất ra. Mỗi dict phần tử phải chứa cấu trúc:
            {
                "case_id": "case_4101",
                "prediction": "A_WIN",
                "case_evidence": ["case_4101_chunk_3"],
                "law_evidence": [{"law_id": "91/2015/QH13", "aid": 15}, ...],
                "api_calls": 8  # Tổng số lượt gọi API đo từ log hệ thống của Khoa
            }

    Returns:
        tuple: Bộ kết quả trả về gồm 2 thành phần:
            - report (dict): Từ điển chứa điểm số Final Score tổng quy đổi và chi tiết 
              từng cấu phần điểm thành phần cùng các thống kê tầng trích xuất liên quan.
              Cấu trúc từ điển đầu ra:
              {
                  "ALQAC_Final_Score": 0.8250,
                  "Components": {
                      "Outcome_Accuracy_70%": 0.9000,
                      "Penalized_Case_Recall_20%": 0.6500,
                      "Micro_Law_F1_10%": 0.6500
                  },
                  "Retrieval_Stats": {"Law_Precision": 0.7000, "Law_Recall": 0.6061}
              }
            - cm_df (pd.DataFrame): Ma trận nhầm lẫn (Confusion Matrix) được định 
              dạng dưới dạng Pandas DataFrame với cả hai trục Chỉ mục (Index) và Cột 
              (Columns) được cố định theo đúng thứ tự 4 nhãn tiêu chuẩn của cuộc thi.

    Toán học và Xử lý Ca cá biệt (Edge Cases):
        - Nếu tập bằng chứng chuẩn rỗng (|G_i^case| = 0): Nếu hệ thống không nhặt thừa 
          (|P_i^case| = 0) -> Recall = 1.0; ngược lại nếu nhặt thừa -> Recall = 0.0.
        - Định nghĩa hằng số phạt n_i: Khớp chính xác quy chế BTC, n_i chính là số lượng 
          bằng chứng đúng thực tế của case đó (|G_i^case|), không phải tổng số phân đoạn 
          của văn bản gốc.
        - Hệ số phạt E_i: Tính theo biểu thức: max(0, 1 - max(0, c_i - 2*n_i) / (3*n_i)). 
          Nếu n_i = 0 (case không cần bằng chứng), gọi API c_i > 0 sẽ bị phạt E_i = 0.0; 
          nếu c_i = 0 -> E_i = 1.0 (Miễn phạt). Hệ số suy giảm dần và chạm 0 tại c_i = 5*n_i.
        - Micro Law F1: Gom tất cả các cặp (law_id, aid) thành chuỗi băm duy nhất, thực 
          hiện phép hợp (Union) trên toàn cục rồi mới tính F1. Trường aid bắt buộc phải 
          là mã số bài viết kiểu số nguyên (int) trích xuất từ corpus_law_pub.json.
    """
    gold_dict = {str(item['case_id']): item for item in gold_data}
    pred_dict = {str(item['case_id']): item for item in pred_data}
    
    total_cases = len(gold_dict)
    correct_outcomes = 0
    total_penalized_case_recall = 0.0
    
    # Siêu tập hợp toàn cục phục vụ Micro Law F1 (Unique global laws over full test set)
    G_law_set = set()
    P_law_set = set()
    
    y_true_list = []
    y_pred_list = []
    
    for case_id, gold in gold_dict.items():
        if case_id not in pred_dict:
            print(f"❌ THIẾU CASE_ID: {case_id} trong file dự đoán!")
            continue
            
        pred = pred_dict[case_id]
        
        # Làm sạch và thu thập nhãn phán quyết
        gt_label = str(gold.get('prediction', '')).strip()
        pred_label = str(pred.get('prediction', '')).strip()
        y_true_list.append(gt_label)
        y_pred_list.append(pred_label)
        
        # 1. OUTCOME ACCURACY (70%)
        if gt_label == pred_label:
            correct_outcomes += 1
            
        # 2. PENALIZED CASE EVIDENCE RECALL (20%)
        # Tự động De-duplicated bằng cấu trúc set theo đúng quy định của BTC
        g_case = {str(seg).strip() for seg in gold.get('case_evidence', [])}
        p_case = {str(seg).strip() for seg in pred.get('case_evidence', [])}
        
        # Tính toán Recall nền tảng chống lỗi tập dữ liệu rỗng
        if len(g_case) == 0:
            case_recall = 1.0 if len(p_case) == 0 else 0.0
        else:
            case_recall = len(p_case.intersection(g_case)) / len(g_case)
        
        # KHỚP RULE BTC: n_i chính là số lượng bằng chứng đúng của case đó (|G_i^case|)
        n_i = len(g_case)
        c_i = pred.get('api_calls', 0)  # Số lượt gọi API đo từ log hệ thống của Khoa
        
        if n_i == 0:
            E_i = 1.0 if c_i == 0 else 0.0
        else:
            # Công thức BTC: E_i = max(0, 1 - max(0, c_i - 2*n_i) / (3*n_i))
            penalty_numerator = max(0, c_i - 2 * n_i)
            penalty_denominator = 3 * n_i
            E_i = max(0.0, 1.0 - (penalty_numerator / penalty_denominator))
            
        total_penalized_case_recall += (case_recall * E_i)
        
        # 3. MICRO LAW EVIDENCE F1 (10%)
        def extract_global_law_keys(evidence_list):
            # aid bắt buộc là mã số bài viết kiểu số nguyên (int) trích xuất từ corpus_law_pub.json
            return {
                f"{str(e.get('law_id')).strip()}_{int(e.get('aid'))}" 
                for e in evidence_list 
                if isinstance(e, dict) and 'law_id' in e and 'aid' in e
            }
            
        G_law_set.update(extract_global_law_keys(gold.get('law_evidence', [])))
        P_law_set.update(extract_global_law_keys(pred.get('law_evidence', [])))

    # --- TÍNH TOÁN CÁC CHỈ SỐ TOÀN CỤC ---
    accuracy = correct_outcomes / total_cases if total_cases > 0 else 0.0
    avg_penalized_case_recall = total_penalized_case_recall / total_cases if total_cases > 0 else 0.0
    
    # Phép toán tập hợp Micro F1 toàn cục (Global Union) theo đồ thị BTC
    tp_law = len(P_law_set.intersection(G_law_set))
    micro_precision = tp_law / len(P_law_set) if len(P_law_set) > 0 else 0.0
    micro_recall = tp_law / len(G_law_set) if len(G_law_set) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # CÔNG THỨC ĐIỂM FINAL CHÍNH THỨC CỦA BTC
    final_score = (0.70 * accuracy) + (0.20 * avg_penalized_case_recall) + (0.10 * micro_f1)
    
    # Dựng ma trận nhầm lẫn cố định theo danh mục nhãn phục vụ chẩn đoán cho Thịnh
    cm_df = _build_confusion_matrix(y_true_list, y_pred_list, ALQAC_LABELS)
    
    # Đóng gói báo cáo thực nghiệm sạch sẽ phục vụ ghi log tự động của Khoa
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
