# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV**: 2A202603955
- **Họ và tên**: Nguyễn Xuân Dương
- **SEED** (in ở ô đầu notebook): 1694

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☒ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☒ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☒ |

## 2. Kết quả chính

 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.778      0.875        0.919   
2. BoW + MLP (deep learning)         0.978   0.909      0.942        0.919   
3a. Encoder (MiniLM) + kNN           0.847   0.949      0.895        0.950   
3b. Encoder (MiniLM) + LogReg        0.979   0.939      0.959        1.000   
3c. Encoder (MiniLM) + MLP head      0.959   0.949      0.954        1.000   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.250      47627.309       0.041   
2. BoW + MLP (deep learning)         3.875      57074.233       0.184   
3a. Encoder (MiniLM) + kNN           4.468        137.676      12.793   
3b. Encoder (MiniLM) + LogReg        4.857       1775.617       6.558   
3c. Encoder (MiniLM) + MLP head      5.607       2299.753       6.690   
4a. LLM nhỏ zero-shot                0.000          1.011     942.429   
4b. LLM lớn zero-shot                0.000          0.863    1099.611   
4c. LLM nhỏ few-shot                 0.000          0.986     824.150   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.060         0.000  
2. BoW + MLP (deep learning)          0.245         0.000  
3a. Encoder (MiniLM) + kNN           14.872         0.000  
3b. Encoder (MiniLM) + LogReg         7.379         0.000  
3c. Encoder (MiniLM) + MLP head       7.724         0.000  
4a. LLM nhỏ zero-shot              1587.684         0.008  
4b. LLM lớn zero-shot              1663.056         0.076  
4c. LLM nhỏ few-shot               1590.015         0.020  

## 3. Dự đoán và thực tế

1. LLM (llm_models): Em dự đoán model lớn có F1 cao hơn model nhỏ. Thực tế cả hai đều có F1 = 0 do 40/40 phản hồi không hợp lệ. Nguyên nhân là định dạng đầu ra chưa được xử lý đúng.

2. Tập 50 case khó (hard_eval): Em dự đoán Encoder đạt Accuracy 75–85%, nhưng thực tế chỉ đạt 66–70%. Điều này cho thấy dữ liệu khó làm giảm đáng kể hiệu quả mô hình.

3. Few-shot (llm_fewshot): Em dự đoán F1 tăng nhẹ khi thêm ví dụ vào prompt, nhưng thực tế F1 vẫn bằng 0 do cả 40 phản hồi không hợp lệ. Điều này cho thấy prompt tốt chưa đủ, cần xử lý đầu ra chính xác.

Bài học: Em nhận ra dự đoán lý thuyết có thể khác xa thực nghiệm. Cần đánh giá cả chất lượng mô hình, định dạng đầu ra và khả năng xử lý dữ liệu khó

## 4. Phân tích 50 case khó

BoW + Logistic Regression có kết quả thấp nhất với Accuracy 52%, đặc biệt gặp khó khăn ở nhóm natural_spam, manipulation và obfuscation. Encoder + MLP đạt kết quả tốt nhất với Accuracy 70%.

Hai trường hợp tiêu biểu:

Case #14: Tin nhắn lừa đảo việc làm tại nhà nhưng sử dụng cách diễn đạt tự nhiên, khiến mô hình khó nhận diện ý đồ spam.

Case #16: Tin nhắn giả dạng người quen bán sản phẩm, sử dụng ngôn ngữ thân thiện nên dễ bị nhầm với tin nhắn hợp lệ.

Nguyên nhân chính là mô hình dựa nhiều vào từ khóa hoặc chưa hiểu đầy đủ ngữ cảnh và ý định của người gửi.

## 5. Thử thách cá nhân

Em sử dụng mô hình Logistic Regression, thử nghiệm ba ngưỡng và đánh giá bằng Precision, Recall, F1-score.

Kết quả cho thấy ngưỡng thấp giúp tăng Recall nhưng có thể giảm Precision. Ngược lại, ngưỡng cao giúp hạn chế chặn nhầm tin nhắn hợp lệ nhưng có thể bỏ sót spam.

Kết luận: Ngân hàng nên ưu tiên ngưỡng 0.8 để hạn chế chặn nhầm tin nhắn OTP.

## 6. Phương án cho sản phẩm thật

Em lựa chọn MiniLM + MLP vì đạt F1 Test = 0.954, Accuracy trên 50 case khó đạt 70% và thuộc nhóm mô hình đáp ứng SLA p95 < 50 ms/tin.

Đề xuất kết hợp Router: mô hình Embedding xử lý tin nhắn thông thường, những tin có xác suất từ 0.3 đến 0.7 được chuyển sang LLM kiểm tra.

Nếu sử dụng LLM nhỏ cho toàn bộ 10 triệu SMS/ngày, chi phí ước tính khoảng 2.250 USD/tháng theo benchmark ban đầu. Vì vậy, Router giúp giảm chi phí và độ trễ, nhưng phải chấp nhận hệ thống phức tạp hơn

## 7. AI log và điều đã học
Trong quá trình thực hiện bài thực hành, em đã sử dụng AI để hỗ trợ giải quyết một số vấn đề:

Sửa lỗi Logistic Regression: AI giải thích nguyên nhân biến lr_model chưa được khai báo. Em đã khởi tạo và huấn luyện lại mô hình.

Thay đổi Threshold: AI hướng dẫn tính Precision, Recall và F1-score tại các ngưỡng 0.2, 0.5 và 0.8. Em thực hiện chạy thử và so sánh kết quả.

Phân tích case khó: AI hỗ trợ giải thích nguyên nhân mô hình phân loại sai ở các tin nhắn có cách diễn đạt tự nhiên hoặc gây nhầm lẫn. Em đối chiếu với kết quả thực nghiệm.

Thiết kế Router: AI đề xuất kết hợp Embedding và LLM để cân bằng tốc độ, độ chính xác và chi phí. Em sử dụng ý tưởng này để đề xuất phương án triển khai thực tế.

Qua bài thực hành, em hiểu rằng mô hình phức tạp hơn không phải lúc nào cũng cho kết quả tốt hơn. Việc lựa chọn mô hình cần cân bằng giữa độ chính xác, tốc độ xử lý và chi phí. Ngoài ra, dữ liệu khó có thể khiến mô hình hoạt động kém hơn đáng kể so với tập kiểm thử thông thường.
