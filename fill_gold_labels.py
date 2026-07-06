import json
import re
from pathlib import Path

def parse_law_provisions(raw_law_str):
    """
    Hàm xử lý chuỗi văn bản pháp luật thô từ BTC thành định dạng mảng JSON.
    Ví dụ: "Bộ luật Dân sự năm 2015 | Điều 584\n" -> [{"law_id": "Bộ luật Dân sự năm 2015", "aid": 584}]
    """
    if not raw_law_str:
        return []
    
    law_evidence = []
    # Tách các điều luật bằng dấu xuống dòng
    lines = raw_law_str.strip().split('\n')
    for line in lines:
        if '|' in line:
            parts = line.split('|')
            law_id = parts[0].strip()
            article_part = parts[1].strip()
            
            # Dùng Regex để tìm tất cả các chữ số (số điều luật) trong chuỗi
            digits = re.findall(r'\d+', article_part)
            if digits:
                aid = int(digits[0])
                law_evidence.append({
                    "law_id": law_id,
                    "aid": aid
                })
    return law_evidence

def auto_fill_gold_labels():
    # 1. Định nghĩa đường dẫn tới các file (dựa trên cấu trúc thư mục của nhóm)
    btc_file_path = Path(r"D:\ALQAC2026\alqac-2026-rag\data\ALQAC2026_public_test.json")
    gold_file_path = Path(r"D:\ALQAC2026\alqac-2026-rag\data\local_validation_gold.json")
    
    # Kiểm tra xem các file có tồn tại không
    if not btc_file_path.exists():
        print(f"❌ Không tìm thấy file của BTC tại: {btc_file_path}")
        return
    if not gold_file_path.exists():
        print(f"❌ Không tìm thấy file gold của bạn tại: {gold_file_path}")
        return

    # 2. Đọc dữ liệu từ file của Ban tổ chức
    print(f"📖 Đang đọc dữ liệu từ file BTC: {btc_file_path}...")
    with open(btc_file_path, "r", encoding="utf-8") as f:
        btc_data = json.load(f)
        
    # Tạo một từ điển để tra cứu nhanh bằng case_id
    btc_lookup = {item["case_id"]: item for item in btc_data if "case_id" in item}

    # 3. Đọc file local_validation_gold.json hiện tại của bạn
    with open(gold_file_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # 4. Duyệt qua từng case và tự động điền nhãn chuẩn
    filled_count = 0
    for item in gold_data:
        case_id = item.get("case_id")
        
        # Nếu case_id này có tồn tại trong file của BTC cung cấp
        if case_id in btc_lookup:
            btc_case = btc_lookup[case_id]
            
            # Điền nhãn prediction (lấy từ verdict_label của BTC)
            if "verdict_label" in btc_case:
                item["prediction"] = btc_case["verdict_label"]
            
            # Điền mảng law_evidence (phân tích từ related_law_provisions của BTC)
            if "related_law_provisions" in btc_case:
                item["law_evidence"] = parse_law_provisions(btc_case["related_law_provisions"])
                
            # Đưa case_evidence về mảng rỗng để không bị lỗi định dạng TODO
            item["case_evidence"] = []
            
            filled_count += 1

    # 5. Ghi đè trực tiếp kết quả trở lại file local_validation_gold.json
    with open(gold_file_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 Thành công! Đã tự động dán nhãn chuẩn cho {filled_count}/{len(gold_data)} cases vào file {gold_file_path}")

if __name__ == "__main__":
    auto_fill_gold_labels()