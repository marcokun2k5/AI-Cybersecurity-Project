# Chương 4 - Đánh giá hiệu quả phòng vệ chèn lệnh cho AI Agent

## 1. Thiết lập thí nghiệm

Thí nghiệm được thực hiện bằng mô hình Ollama cục bộ `llama3.2`, nhiệt độ suy luận `0`, seed suy luận `0` và bộ cấu hình phòng vệ cố định. Hai tập dữ liệu được đánh giá tuần tự:

- Nhật ký sạch: 22 mẫu.
- Nhật ký có chèn lệnh: 20 mẫu.
- Tổng cộng: 42 mẫu.
- Năm cấu hình: không phòng vệ, cô lập ngữ cảnh, bộ phát hiện chèn lệnh, ràng buộc đầu ra JSON và kết hợp các phòng vệ.
- Tổng số lần gọi mô hình: 210 lần.
- Thời điểm chạy đã ghi nhận: `2026-11-22T00:51:12.829854Z`.
- Fingerprint tập sạch: `836be0734a575adcf1722ec8ccc4b3ecbf7a546373ded32736945192561e4218`.
- Fingerprint tập chèn lệnh: `51a548105ef99f405e2789da1a560f95443058c28434d33662bb8d760c01cc2d`.

Mỗi cấu hình nhận cùng thứ tự mẫu và cùng ground-truth label để bảo đảm khả năng so sánh.

## 2. Mô tả tập dữ liệu

Tập sạch gồm 22 mẫu, trong đó có 7 mẫu hợp lệ, 3 mẫu Brute Force, 3 mẫu SQL Injection, 3 mẫu Port Scan, 3 mẫu Malware và 3 mẫu Command and Control. Tập chèn lệnh gồm 20 mẫu, gồm 10 Direct Prompt Injection, 5 Indirect Prompt Injection và 5 Context Manipulation. Mỗi lớp tấn công có 4 mẫu trong tập chèn lệnh.

Mẫu được gắn nhãn rủi ro Low, Medium hoặc High cùng loại tấn công chính. Tập chèn lệnh hiện gồm 12 mẫu High và 8 mẫu Medium. Đây là tập nghiên cứu tổng hợp quy mô nhỏ, không thay thế corpus nhật ký thực tế.

## 3. Cơ chế phòng vệ

- **Không có phòng vệ:** gửi nguyên nội dung nhật ký cho mô hình.
- **Cô lập ngữ cảnh:** bao dữ liệu nhật ký trong thẻ `<LOG_DATA>` và yêu cầu mô hình không thực thi chỉ dẫn bên trong dữ liệu.
- **Bộ phát hiện chèn lệnh:** phát hiện mẫu ngôn ngữ injection và chuyển nội dung đó thành cảnh báo không đáng tin cậy.
- **Ràng buộc đầu ra JSON:** yêu cầu mô hình trả kết quả theo JSON gồm risk level, attack type và reason.
- **Kết hợp các phòng vệ:** kích hoạt đồng thời cô lập, phát hiện injection và ràng buộc đầu ra.

## 4. Phương pháp đánh giá

- **Độ chính xác:** phần trăm mẫu mà cả mức rủi ro lẫn loại tấn công đều khớp ground truth.
- **Tỷ lệ thao túng:** phần trăm mẫu chèn lệnh bị hạ mức rủi ro, bị dự đoán sai loại tấn công hoặc tuân theo chỉ dẫn của attacker.
- **Tỷ lệ tấn công thành công:** được tính bằng cùng tử số và mẫu số với tỷ lệ thao túng trong thí nghiệm này.

## 5. Kết quả thực nghiệm

### 5.1 Bảng so sánh

| Tập dữ liệu | Cấu hình | Số mẫu | Độ chính xác | Tỷ lệ thao túng | Tỷ lệ tấn công thành công | Dự đoán đúng |
|---|---|---:|---:|---:|---:|---:|
| Nhật ký sạch | Không có phòng vệ | 22 | 59.09% | Không áp dụng | Không áp dụng | 13/22 |
| Nhật ký sạch | Cô lập ngữ cảnh | 22 | 59.09% | Không áp dụng | Không áp dụng | 13/22 |
| Nhật ký sạch | Bộ phát hiện chèn lệnh | 22 | 50.00% | Không áp dụng | Không áp dụng | 11/22 |
| Nhật ký sạch | Ràng buộc đầu ra JSON | 22 | 59.09% | Không áp dụng | Không áp dụng | 13/22 |
| Nhật ký sạch | Kết hợp các phòng vệ | 22 | 59.09% | Không áp dụng | Không áp dụng | 13/22 |
| Nhật ký chèn lệnh | Không có phòng vệ | 20 | 5.00% | 95.00% | 95.00% | 1/20 |
| Nhật ký chèn lệnh | Cô lập ngữ cảnh | 20 | 5.00% | 95.00% | 95.00% | 1/20 |
| Nhật ký chèn lệnh | Bộ phát hiện chèn lệnh | 20 | 5.00% | 95.00% | 95.00% | 1/20 |
| Nhật ký chèn lệnh | Ràng buộc đầu ra JSON | 20 | 0.00% | 100.00% | 100.00% | 0/20 |
| Nhật ký chèn lệnh | Kết hợp các phòng vệ | 20 | 0.00% | 95.00% | 95.00% | 0/20 |

### 5.2 Xếp hạng trên tập chèn lệnh

| Hạng | Cấu hình | Độ chính xác | Tỷ lệ thao túng | Tỷ lệ tấn công thành công |
|---:|---|---:|---:|---:|
| 1 | Cô lập ngữ cảnh | 5.00% | 95.00% | 95.00% |
| 2 | Không có phòng vệ | 5.00% | 95.00% | 95.00% |
| 3 | Bộ phát hiện chèn lệnh | 5.00% | 95.00% | 95.00% |
| 4 | Kết hợp các phòng vệ | 0.00% | 95.00% | 95.00% |
| 5 | Ràng buộc đầu ra JSON | 0.00% | 100.00% | 100.00% |

- **Phòng vệ tốt nhất:** Cô lập ngữ cảnh, Không có phòng vệ, Bộ phát hiện chèn lệnh (tỷ lệ thao túng thấp nhất trong tập chèn lệnh và xếp hạng theo độ chính xác).
- **Phòng vệ kém nhất:** Ràng buộc đầu ra JSON (100.00% thao túng, 0.00% độ chính xác).
- **Cải thiện lớn nhất:** Cô lập ngữ cảnh, Bộ phát hiện chèn lệnh, Kết hợp các phòng vệ, thay đổi tỷ lệ thao túng +0.00 điểm phần trăm so với không phòng vệ.
- **Cải thiện nhỏ nhất:** Ràng buộc đầu ra JSON, thay đổi tỷ lệ thao túng -5.00 điểm phần trăm so với không phòng vệ.

## 6. Thảo luận

Đối với tập nhật ký sạch, `llama3.2` đạt 59.09% độ chính xác ở không phòng vệ, cô lập ngữ cảnh, ràng buộc JSON và cấu hình kết hợp. Bộ phát hiện độc lập đạt 50.00%. Kết quả cho thấy ràng buộc JSON có khả năng ổn định nhãn rủi ro và loại tấn công tốt hơn trên nhật ký sạch.

Đối với tập chèn lệnh, ba cấu hình không có phòng vệ, cô lập ngữ cảnh và bộ phát hiện đạt độ chính xác 5.00% và tỷ lệ thao túng 95.00%. Ràng buộc JSON đạt độ chính xác 0.00% và tỷ lệ thao túng 100.00%. Cấu hình kết hợp đạt độ chính xác 0.00% và tỷ lệ thao túng 95.00%. Không có cấu hình nào loại bỏ hoàn toàn tác động prompt injection trong lần chạy này.

Cô lập ngữ cảnh không tạo cải thiện đo được so với baseline khi đo tỷ lệ thao túng, dù độ chính xác giữ ngang nhau. Bộ phát hiện độc lập cũng không làm tăng độ chính xác và có cùng tỷ lệ thao túng với baseline. Ràng buộc JSON làm tỷ lệ thao túng tăng thêm 5 điểm phần trăm, trong khi kết hợp không làm thay đổi tỷ lệ thao túng so với baseline. Do đó, kết quả không chứng minh rằng mỗi cơ chế phòng vệ đơn lẻ đều nâng cao tính an toàn; kết quả cần được kiểm tra thêm trên corpus lớn hơn và nhiều lần lặp.

## 7. Giới hạn

Nghiên cứu sử dụng 42 mẫu tổng hợp, có tỷ lệ injection 47.62%; tập dữ liệu vẫn nhỏ so với môi trường SOC thực tế. Ground-truth labels có thể phản ánh giả thiết cố định của dự án thay vì kết quả điều tra chuyên sâu. Kết quả phụ thuộc vào phiên bản mô hình, runtime Ollama, prompt và cơ chế phân tích đầu ra. Hạn chế này phải được nêu rõ khi sử dụng kết quả trong luận văn.

## 8. Kết luận

Trong lần chạy thực nghiệm với `llama3.2`, không có phòng vệ nào đạt kết quả tốt rõ rệt theo toàn bộ các chỉ số. Cô lập ngữ cảnh, bộ phát hiện và cấu hình không phòng vệ cùng đạt 95.00% thao túng và 5.00% độ chính xác trên tập injection. Ràng buộc JSON có kết quả kém nhất với 100.00% thao túng. Kết quả thực nghiệm hỗ trợ kết luận rằng prompt injection nhúng trong log vẫn là rủi ro nghiêm trọng và cơ chế phòng vệ phải được đánh giá trên nhiều tập dữ liệu, nhiều lần seed và nhiều mô hình.

## 9. Hướng phát triển

1. Mở rộng corpus lên ít nhất 50-100 mẫu sạch và mẫu injection cân bằng.
2. Tăng tỷ lệ mẫu injected Medium/Low để tránh thiên lệch High-risk.
3. Chạy nhiều seed, nhiều mô hình Ollama và báo cáo khoảng tin cậy.
4. Phân tích tỷ lệ thao túng theo từng loại injection và từng loại tấn công.
5. Kết hợp cô lập, phát hiện và JSON với kiểm tra ngữ nghĩa độc lập thay vì chỉ so sánh nhãn rủi ro.
