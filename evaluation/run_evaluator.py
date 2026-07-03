# evaluation/run_evaluator.py
import json
import os
import sys

# Tự động thêm thư mục cha (thư mục gốc dự án) vào hệ thống tra cứu module của Python
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

from metrics import evaluate_alqac_system

def load_json_data(file_path):
    """Nạp dữ liệu JSON, hỗ trợ kiểm tra file tồn tại."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Không tìm thấy tệp dữ liệu tại: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"❌ Tệp {file_path} không đúng định dạng cấu trúc JSON!")

def run_local_evaluation(gold_path, pred_path, report_output_path):
    print("🚀 Đang khởi động bộ đánh giá Local cho hệ thống ALQAC...")
    
    # Đọc dữ liệu
    gold_data = load_json_data(gold_path)
    pred_data = load_json_data(pred_path)
    
    # Thực thi chấm điểm
    report, cm_df = evaluate_alqac_system(gold_data, pred_data)
    
    print("\n" + "="*50)
    print("📊 BÁO CÁO THỰC NGHIỆM")
    print("="*50)
    print(f"🎯 ALQAC FINAL SCORE          : {report['ALQAC_Final_Score']:.4f}")
    print(f"├─ Outcome Accuracy (70%)     : {report['Components']['Outcome_Accuracy_70%']:.4f}")
    print(f"├─ Penalized Case Recall (20%): {report['Components']['Penalized_Case_Recall_20%']:.4f}")
    print(f"└─ Micro Law Evidence F1 (10%): {report['Components']['Micro_Law_F1_10%']:.4f}")
    print("-" * 50)
    print(f"📈 Thống kê tầng Luật (Micro):")
    print(f"├─ Precision                  : {report['Retrieval_Stats']['Law_Precision']:.4f}")
    print(f"└─ Recall                     : {report['Retrieval_Stats']['Law_Recall']:.4f}")
    print("="*50)
    
    print("\n🧩 MA TRẬN NHẦM LẪN (CONFUSION MATRIX DIAGNOSTICS):")
    print(cm_df)
    print("="*50 + "\n")
    
    # Lưu báo cáo
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"💾 Báo cáo thực nghiệm đã được lưu tại: {report_output_path}\n")

if __name__ == "__main__":
    # Sử dụng os.path.join kết hợp PROJECT_ROOT để cố định đường dẫn tuyệt đối chuẩn xác
    GOLD_FILE = os.path.join(PROJECT_ROOT, "data", "local_validation_gold.json")
    PRED_FILE = os.path.join(PROJECT_ROOT, "experiments", "run_v0_baseline.json")
    REPORT_FILE = os.path.join(PROJECT_ROOT, "experiments", "baseline_report_v0.json")
    
    run_local_evaluation(GOLD_FILE, PRED_FILE, REPORT_FILE)