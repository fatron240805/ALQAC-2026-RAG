# evaluation/retrieval_eval.py
import json
import os
import sys

# Cấu hình đường dẫn tuyệt đối cho hệ thống repo
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

def evaluate_retrieval_only(gold_data, pred_data):
    """Đánh giá độc lập tầng Retrieval (Trích xuất dữ liệu) cho hệ thống Agentic RAG.

    Hàm này bóc tách riêng cấu phần trích xuất Tình tiết vụ án (Case Content - chiếm 20% 
    điểm tổng) và cấu phần trích xuất Văn bản pháp luật (Law Corpus - chiếm 10% điểm tổng) 
    để tính toán chuyên sâu các chỉ số Recall, Precision, và F1. Điều này giúp Retrieval 
    Engineer (Hưng) tối ưu hóa các tham số Hyperparameters của Vector DB (như chunk size, 
    overlap, top_k, similarity threshold, reranker alpha) một cách độc lập mà không cần 
    chạy qua tầng suy luận LLM của Thịnh.

    Cơ chế đo lường & Phạt API Hiệu năng:
        - Case Content Retrieval: Tính toán chỉ số Recall truyền thống trên từng case. Tuy 
          nhiên, chỉ số này sẽ nhân với Hệ số phạt API (API Efficiency Factor E_i) mô phỏng 
          theo log server của BTC.
          Biểu thức toán học của hệ số phạt: E_i = max(0, 1 - max(0, c_i - 2*n_i) / (3*n_i))
          Trong đó:
            + c_i: Số lượt gọi API nội dung vụ án thực tế của hệ thống.
            + n_i: Số lượng phân đoạn (chunks) chứng cứ đúng trong đáp án chuẩn (|G_i^case|).
          Hệ thống sẽ không bị phạt nếu c_i <= 2*n_i. Nếu vượt quá, điểm số của case sẽ giảm 
          dần về 0 tại ngưỡng c_i = 5*n_i (Spam chọc API liên tục).
        - Law Corpus Retrieval: Tính toán dựa trên nguyên lý Global Union (Hợp toàn cục). 
          Tất cả các thực thể luật đúng (Gold) và đoán (Pred) trên toàn bộ tập test sẽ được 
          gom vào hai rổ lớn duy nhất để tính một điểm Micro F1 chung cho toàn hệ thống, 
          khớp chính xác đồ thị ma trận của BTC.

    Args:
        gold_data (list of dict): Tập dữ liệu nhãn chuẩn (Ground Truth nội bộ). 
            Mỗi phần tử bắt buộc phải có cấu trúc:
            {
                "case_id": "case_4101",
                "case_evidence": ["case_4101_chunk_3", "case_4101_chunk_5"],
                "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}, ...]
            }
        pred_data (list of dict): Tập dữ liệu log chạy đầu ra từ module Retrieval của Hưng. 
            Mỗi phần tử bắt buộc phải có cấu trúc:
            {
                "case_id": "case_4101",
                "case_evidence": ["case_4101_chunk_3"],
                "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}, ...],
                "api_calls": 8  # Tổng số lượt chọc API thực tế tính trên case này
            }

    Returns:
        dict: Báo cáo phân tích chuyên sâu chất lượng trích xuất cấu trúc 2 tầng dữ liệu sạch:
            {
                "Case_Content_Retrieval": {
                    "Raw_Case_Recall": 0.8500,  # Recall gốc khi chưa áp hệ số phạt API
                    "Penalized_Case_Recall_Official": 0.7250,  # Điểm số cấu phần 20% thực tế
                    "Avg_API_Calls_Per_Case": 4.2,  # Số lượt chọc API trung bình/vụ án
                    "API_Efficiency_Loss": 0.1250  # Mức độ hao hụt điểm số do gọi API quá tay
                },
                "Law_Corpus_Retrieval_Global_Micro": {
                    "Micro_Law_F1_Official": 0.7812,  # Điểm số cấu phần 10% thực tế
                    "Micro_Law_Precision": 0.8000,
                    "Micro_Law_Recall": 0.7632,
                    "Total_Unique_Gold_Laws": 38,  # Tổng số điều luật độc nhất trong tập Gold
                    "Total_Unique_Pred_Laws": 35,  # Tổng số điều luật độc nhất hệ thống nhặt ra
                    "Correct_Unique_Laws_Found": 28  # Số điều luật nhặt chính xác (True Positive)
                }
            }

    Edge Cases Handled:
        - Nếu tập bằng chứng chuẩn rỗng (|G_i^case| = 0): Nếu hệ thống không nhặt thừa 
          (|P_i^case| = 0) -> Recall = 1.0; nếu nhặt thừa -> Recall = 0.0.
        - Nếu n_i = 0 (Case không cần bằng chứng): Gọi API c_i > 0 sẽ bị phạt E_i = 0.0; 
          nếu c_i = 0 -> E_i = 1.0.
        - Tự động `.strip()` và ép kiểu số nguyên `int()` cho trường `aid` bài viết luật.
    """
    gold_dict = {str(item['case_id']): item for item in gold_data}
    pred_dict = {str(item['case_id']): item for item in pred_data}
    
    # -------------------------------------------------------------------------
    # METER 1: CASE EVIDENCE RETRIEVAL (Tình tiết vụ án - Trọng số 20%)
    # -------------------------------------------------------------------------
    total_case_recall = 0.0
    total_penalized_case_recall = 0.0
    total_api_calls = 0
    
    # -------------------------------------------------------------------------
    # METER 2: LAW EVIDENCE RETRIEVAL (Văn bản Pháp luật - Trọng số 10%)
    # -------------------------------------------------------------------------
    # Siêu tập hợp toàn cục thu thập phần tử độc nhất (Unique Global Sets)
    G_law_set = set()
    P_law_set = set()
    
    for case_id, gold in gold_dict.items():
        if case_id not in pred_dict:
            continue
            
        pred = pred_dict[case_id]
        
        # --- Xử lý tầng Tình tiết Vụ án ---
        # Tự động De-duplicated bằng phép toán tập hợp (set) theo quy định của BTC
        g_case = {str(seg).strip() for seg in gold.get('case_evidence', [])}
        p_case = {str(seg).strip() for seg in pred.get('case_evidence', [])}
        
        # Tính Recall nền tảng chống lỗi tập dữ liệu rỗng
        if len(g_case) == 0:
            case_recall = 1.0 if len(p_case) == 0 else 0.0
        else:
            case_recall = len(p_case.intersection(g_case)) / len(g_case)
        total_case_recall += case_recall
        
        # KHỚP RULE BTC: n_i chính là số lượng bằng chứng đúng của case đó (|g_case|)
        n_i = len(g_case)
        c_i = int(pred.get('api_calls', 0))
        total_api_calls += c_i
        
        # Tính toán Hệ số phạt API hiệu năng theo Compact Max Form của BTC
        if n_i == 0:
            E_i = 1.0 if c_i == 0 else 0.0
        else:
            penalty_numerator = max(0, c_i - 2 * n_i)
            penalty_denominator = 3 * n_i
            E_i = max(0.0, 1.0 - (penalty_numerator / penalty_denominator))
            
        total_penalized_case_recall += (case_recall * E_i)
        
        # --- Xử lý tầng Luật (Theo phép toán Hợp toàn cục - Global Union) ---
        def extract_global_law_keys(evidence_list):
            # Cú pháp băm phẳng loại bỏ case_id, ép kiểu int(aid) trích xuất từ corpus_law_pub.json
            return {
                f"{str(e.get('law_id')).strip()}_{int(e.get('aid'))}" 
                for e in evidence_list 
                if isinstance(e, dict) and 'law_id' in e and 'aid' in e
            }
            
        G_law_set.update(extract_global_law_keys(gold.get('law_evidence', [])))
        P_law_set.update(extract_global_law_keys(pred.get('law_evidence', [])))

    # --- Tổng hợp số liệu tầng Case Content ---
    total_cases = len(gold_dict)
    avg_case_recall = total_case_recall / total_cases if total_cases > 0 else 0.0
    avg_penalized_case_recall = total_penalized_case_recall / total_cases if total_cases > 0 else 0.0
    avg_api_calls = total_api_calls / total_cases if total_cases > 0 else 0.0
    
    # --- Tổng hợp số liệu tầng Luật (Micro-averaged Metrics trên siêu tập hợp) ---
    tp_law = len(P_law_set.intersection(G_law_set))
    micro_law_precision = tp_law / len(P_law_set) if len(P_law_set) > 0 else 0.0
    micro_law_recall = tp_law / len(G_law_set) if len(G_law_set) > 0 else 0.0
    micro_law_f1 = (2 * micro_law_precision * micro_law_recall) / (micro_law_precision + micro_law_recall) if (micro_law_precision + micro_law_recall) > 0 else 0.0
    
    # Đóng gói báo cáo phân tích chuyên sâu cho cấu phần trích xuất
    retrieval_report = {
        "Case_Content_Retrieval": {
            "Raw_Case_Recall": round(avg_case_recall, 4),
            "Penalized_Case_Recall_Official": round(avg_penalized_case_recall, 4),
            "Avg_API_Calls_Per_Case": round(avg_api_calls, 1),
            "API_Efficiency_Loss": round(avg_case_recall - avg_penalized_case_recall, 4)
        },
        "Law_Corpus_Retrieval_Global_Micro": {
            "Micro_Law_F1_Official": round(micro_law_f1, 4),
            "Micro_Law_Precision": round(micro_law_precision, 4),
            "Micro_Law_Recall": round(micro_law_recall, 4),
            "Total_Unique_Gold_Laws": len(G_law_set),
            "Total_Unique_Pred_Laws": len(P_law_set),
            "Correct_Unique_Laws_Found": tp_law
        }
    }
    return retrieval_report

if __name__ == "__main__":
    # Đường dẫn tệp phục vụ chạy thử nghiệm trinh sát nhanh tại Local
    GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "local_validation_gold.json")
    PRED_FILE = os.path.join(PROJECT_ROOT, "experiments", "run_v0_baseline.json")
    
    if not os.path.exists(GOLD_FILE) or not os.path.exists(PRED_FILE):
        print("⚠️ Cảnh báo: Vui lòng chạy thiết lập mock data trước để tạo file json chạy thử nghiệm.")
        sys.exit(1)
        
    with open(GOLD_FILE, "r", encoding="utf-8") as g, open(PRED_FILE, "r", encoding="utf-8") as p:
        report = evaluate_retrieval_only(json.load(g), json.load(p))
        
    print("\n" + "*"*15)
    print("📈 BÁO CÁO PHÂN TÍCH CHUYÊN SÂU TẦNG RETRIEVAL")
    print("*"*15)
    print(json.dumps(report, indent=4, ensure_ascii=False))
    print("="*45 + "\n")