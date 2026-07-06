# 📊 ALQAC 2026 - Evaluation & Submission Module

Thư mục này chứa toàn bộ hạ tầng đo lường chỉ số nội bộ (Local Evaluation) và công cụ đóng gói tệp nộp bài (Submission Packaging) của Nhóm 4, mô phỏng chính xác **Luật chơi 70-20-10** của BTC ALQAC 2026 và tích hợp các biện pháp phòng thủ cho vòng Private Test.

---

## 🧭 1. Quy trình Phối hợp Dữ liệu (Double-File Workflow)

Để vừa phục vụ công tác chẩn đoán sâu ở Local (cần tính điểm phạt API, chẩn đoán tầng Retrieval), vừa đảm bảo tệp nộp lên Leaderboard không bị rác dữ liệu dẫn đến **Desk-reject**, quy trình phối hợp giữa **Khoa (Inference)** và **Tuấn Anh (QA/Evaluation)** sẽ vận hành như sau:

```text
       [ Pipeline Agent RAG của Khoa ]
                     │
                     ▼
    Tệp Log Đầy Đủ: experiments/run_v0_baseline.json
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
[ run_evaluator.py ]      [ build_submission.py ]
        │                         │
        ▼                         ▼
 Báo cáo Local 70-20-10    Tệp Sạch: submission.json
 (Dùng để tối ưu RAG)     (Nộp lên Leaderboard BTC)
```

## 📂 2. Cấu trúc và Chức năng các Tệp (File Architecture)
Hệ thống đánh giá được chia thành 4 module độc lập, phân tách rõ ràng vai trò:

- **`metrics.py` (Core Engine):** Trái tim của bộ Evaluator. Chứa các thuật toán tính toán 3 chỉ số cốt lõi: *Outcome Accuracy (70%)*, *Penalized Case Recall (20%)*, và *Micro Law Evidence F1 (10%)*. Đã tích hợp logic chống lỗi chia cho 0 và phép toán Hợp toàn cục (Global Union) chuẩn BTC dựa trên độ dài tập Gold thực tế ($n_i = |G_i^{case}|$).
- **`run_evaluator.py` (System Evaluator):** Script chạy chính dành cho toàn hệ thống. So sánh file `pred` của Khoa với file `gold`, xuất ra báo cáo điểm tổng (Final Score) và Ma trận nhầm lẫn (Confusion Matrix) để Thịnh chẩn đoán lỗi prompt LLM.
- **`retrieval_eval.py` (Retrieval Diagnostics):** Script "nội soi" dành riêng cho **Hưng**. Tách bỏ hoàn toàn tầng suy luận LLM, chỉ tập trung đánh giá hiệu quả của Vector DB/BM25, đo lường điểm hao hụt do gọi API quá tay (`API_Efficiency_Loss`) và thống kê số lượng luật để điều chỉnh `top_k`.
- **`build_submission.py` (Submission Packer):** Công cụ đóng gói an toàn. Tự động đọc file log của Khoa, "gọt" sạch các biến đo lường nội bộ (`api_calls`), chuẩn hóa kiểu dữ liệu số nguyên cho `aid`, và xuất ra file `submission.json` chuẩn format ở thư mục gốc.

## 📋 3. Ràng buộc Định dạng Dữ liệu Đầu vào (Yêu cầu cho Pipeline)
Để bộ Evaluator hoạt động không bị crash, **Khoa** cần cấu hình tệp log đầu ra (`experiments/run_v0_baseline.json`) chứa đầy đủ các trường dữ liệu sau cho mỗi test case:

```json
[
        {
                "case_id": "case_4101",
                "prediction": "A_WIN",
                "case_evidence": ["case_4101_chunk_3"],
                "law_evidence": [
                        {
                                "law_id": "47/2010/QH12",
                                "aid": 270
                        }
                ],
                "api_calls": 5
        }
]
```

> 💡 **Lưu ý quan trọng:** Trường `api_calls` là bắt buộc phải có trong file log nội bộ để hệ thống tính toán hệ số phạt hiệu năng ($E_i$). Trường này sẽ được tự động xóa bỏ hoàn toàn khi chạy script đóng gói nộp bài.

## 🚀 4. Hướng dẫn Chạy Hệ thống (How to Run)
Mở terminal tại thư mục gốc của dự án (`alqac-2026-rag`) và thực thi các lệnh sau tùy theo mục đích:

### 🎯 4.1. Đánh giá toàn diện Hệ thống (End-to-End Evaluation)
Dùng để xem điểm số tổng hợp mô phỏng Leaderboard và Ma trận nhầm lẫn để chẩn đoán nhãn thắng/thua.

```bash
python evaluation/run_evaluator.py
```

- **Đầu vào:** `data/local_validation_gold.json` và `experiments/run_v0_baseline.json`.
- **Đầu ra:** Báo cáo in trực tiếp ra terminal và lưu tệp cấu trúc JSON tại `experiments/baseline_report_v0.json`.

### 🔍 4.2. Chẩn đoán chuyên sâu tầng Tìm kiếm (Dành riêng cho Hưng)
Dùng khi Hưng thay đổi mô hình Embedding, tinh chỉnh `chunk_size`, `overlap` hoặc thử nghiệm thuật toán Fusion/Reranker.

```bash
python evaluation/retrieval_eval.py
```

- **Đầu vào:** `data/local_validation_gold.json` và `experiments/run_v0_baseline.json`.
- **Đầu ra:** Báo cáo chi tiết về hiệu năng trích xuất: `Raw_Case_Recall`, `API_Efficiency_Loss`, và `Micro_Law_Precision/Recall`.

### 📦 4.3. Đóng gói tệp nộp bài chính thức (Build Submission)
Dùng khi chuẩn bị nộp bài lên hệ thống máy chủ Leaderboard của BTC. Chỉ tiến hành nộp khi điểm Evaluator nội bộ có sự tăng trưởng!

```bash
python evaluation/build_submission.py
```

- **Đầu vào:** `experiments/run_v0_baseline.json`.
- **Đầu ra:** Tệp `submission.json` chuẩn 100% định dạng `Expected format` của BTC nằm tại thư mục gốc của dự án.

## ⚠️ 5. Các Cảnh báo Cốt lõi (Vui lòng tuân thủ)

1. **Rate Limit (Khoa chú ý):** Hệ thống API của BTC giới hạn nghiêm ngặt **1 request / 5 giây** cho mỗi đội. Kịch bản gọi API trong `run_pipeline.py` bắt buộc phải cài đặt hàng đợi hoặc cơ chế delay `time.sleep(5)` để không bị block IP.
2. **Submission Limit:** BTC giới hạn tối đa **3 lần nộp bài/ngày**. Tuyệt đối không dùng Leaderboard BTC làm môi trường test thử sai để tránh bị overfitting tập Public. Mọi quyết định tối ưu phải dựa trên la bàn đánh giá Local.
3. **Data Typing cho Luật:** Trường `aid` trong mảng dữ liệu luật bắt buộc phải là số nguyên `int` đại diện cho ID bài viết trong tệp `corpus_law_pub.json`. Tuyệt đối không truyền chuỗi văn bản tự do hoặc vị trí index của mảng.