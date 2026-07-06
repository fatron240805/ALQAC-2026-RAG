# evaluation/build_submission.py
import json
import os

ALQAC_VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}

def convert_to_official_submission(experiment_file_path, submission_output_path):
    """Đọc file thực nghiệm, lọc cấu trúc dữ liệu chuẩn xác 100% theo Expected format của BTC.
    
    Giữ lại case_evidence và law_evidence, loại bỏ các biến đo lường nội bộ (api_calls, n_segments).
    """
    if not os.path.exists(experiment_file_path):
        print(f"❌ LỖI: Không tìm thấy file thực nghiệm tại {experiment_file_path}")
        return

    with open(experiment_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    official_submission_list = []
    malformed_count = 0
    
    for item in raw_data:
        case_id = str(item.get("case_id", "")).strip()
        prediction = str(item.get("prediction", "")).strip()
        
        # Đọc case_evidence (Mặc định trả về mảng rỗng [] nếu không tìm thấy)
        case_evidence = item.get("case_evidence", [])
        if not isinstance(case_evidence, list):
            case_evidence = [str(case_evidence)] if case_evidence else []
        else:
            case_evidence = [str(seg).strip() for seg in case_evidence]
        
        # Kiểm tra tính hợp lệ của nhãn kết quả
        if prediction not in ALQAC_VALID_LABELS:
            print(f"⚠️ CẢNH BÁO: Case {case_id} có nhãn không hợp lệ: '{prediction}'!")
            malformed_count += 1
            
        # Lọc sạch cấu trúc law_evidence
        law_evidence = item.get("law_evidence", [])
        cleaned_laws = []
        for law in law_evidence:
            if isinstance(law, dict) and "law_id" in law and "aid" in law:
                cleaned_laws.append({
                    "law_id": str(law["law_id"]).strip(),
                    "aid": int(law["aid"])  # Ép kiểu int bắt buộc theo quy định BTC
                })
        
        # Đóng gói đúng cấu trúc 4 trường bắt buộc trên giao diện web BTC
        official_item = {
            "case_id": case_id,
            "prediction": prediction,
            "case_evidence": case_evidence,
            "law_evidence": cleaned_laws
        }
        official_submission_list.append(official_item)
        
    # Xuất file submission.json sạch tại thư mục gốc
    with open(submission_output_path, "w", encoding="utf-8") as f:
        json.dump(official_submission_list, f, ensure_ascii=False, indent=2)
        
    print("\n" + "🏁"*20)
    print(f"🎉 ĐÃ ĐÓNG GÓI THÀNH CÔNG TỆP SUBMISSION CHUẨN BTC: {submission_output_path}")
    print(f"├─ Tổng số vụ án đã đóng gói: {len(official_submission_list)}")
    print(f"└─ Số nhãn lỗi phát hiện     : {malformed_count}")
    print("🏁"*20 + "\n")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    EXP_FILE = os.path.join(PROJECT_ROOT, "experiments", "run_v0_baseline.json")
    SUB_FILE = os.path.join(PROJECT_ROOT, "submission.json")
    
    convert_to_official_submission(EXP_FILE, SUB_FILE)