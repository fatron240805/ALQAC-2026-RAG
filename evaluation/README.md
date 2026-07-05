# 📊 ALQAC 2026 - Evaluation & Submission Module

Thư mục này chứa toàn bộ hạ tầng đo lường chỉ số nội bộ (Local Evaluation) và công cụ đóng gói tệp nộp bài (Submission Packaging) của Nhóm 4, mô phỏng chính xác **Luật chơi 70-20-10** của BTC ALQAC 2026.

---

## 🧭 1. Quy trình Phối hợp Dữ liệu (Double-File Workflow)

Để vừa phục vụ công tác chẩn đoán sâu ở Local (cần tính điểm phạt API, ma trận nhầm lẫn nhãn), vừa đảm bảo tệp nộp lên Leaderboard không bị rác dữ liệu dẫn đến **Desk-reject**, quy trình phối hợp giữa **Khoa (Inference)** và **Tuấn Anh (QA/Evaluation)** sẽ vận hành như sau:

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