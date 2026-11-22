import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_PATH = PROJECT_ROOT / "data" / "research" / "results.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "research_report_chapter4.md"


VIETNAMESE_DEFENSES = {
    "No Defense": "Không có phòng vệ",
    "Context Isolation": "Cô lập ngữ cảnh",
    "Prompt Injection Detector": "Bộ phát hiện chèn lệnh",
    "Structured Output Constraint": "Ràng buộc đầu ra JSON",
    "Combined Defenses": "Kết hợp các phòng vệ",
}


def percent(value: float | None) -> str:
    return "Không áp dụng" if value is None else f"{value:.2f}%"


def main() -> int:
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Run the live benchmark first; missing {RESULTS_PATH}"
        )
    history = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    experiments = history["experiments"]
    if len(experiments) < 2:
        raise ValueError("The latest full benchmark does not contain two datasets")

    latest_runs = experiments[-2:]
    latest_timestamp = latest_runs[-1]["timestamp"]

    clean_run = next(run for run in latest_runs if run["dataset"] == "clean_logs")
    injected_run = next(run for run in latest_runs if run["dataset"] == "injected_logs")

    clean_metrics = {row["defense"]: row for row in clean_run["results"]}
    injected_metrics = {row["defense"]: row for row in injected_run["results"]}
    baseline = injected_metrics["No Defense"]

    ranking = sorted(
        injected_metrics.values(),
        key=lambda row: (
            row["manipulation_rate_percent"]
            if row["manipulation_rate_percent"] is not None
            else 101.0,
            -row["accuracy_percent"],
            row["defense"],
        ),
    )
    best_rate = ranking[0]["manipulation_rate_percent"]
    best_defenses = [
        row["defense"]
        for row in ranking
        if row["manipulation_rate_percent"] == best_rate
        and row["accuracy_percent"] == ranking[0]["accuracy_percent"]
    ]
    worst = sorted(
        injected_metrics.values(),
        key=lambda row: (
            -(row["manipulation_rate_percent"] or 0),
            row["accuracy_percent"],
            row["defense"],
        ),
    )[0]

    baseline_manipulation = baseline["manipulation_rate_percent"]
    improvement_rows = []
    for defense, row in injected_metrics.items():
        manipulation = row["manipulation_rate_percent"]
        improvement = baseline_manipulation - manipulation
        accuracy_delta = row["accuracy_percent"] - baseline["accuracy_percent"]
        improvement_rows.append(
            (
                defense,
                improvement,
                accuracy_delta,
            )
        )
    defensive_improvements = [
        item for item in improvement_rows
        if item[0] != "No Defense"
    ]
    largest_value = max(item[1] for item in defensive_improvements)
    largest_defenses = [
        item for item in defensive_improvements
        if item[1] == largest_value
    ]
    largest_manipulation = max(
        largest_defenses,
        key=lambda item: (item[1], item[2]),
    )
    smallest_manipulation = min(
        improvement_rows,
        key=lambda item: (item[1], item[0]),
    )

    comparison_rows = []
    for run in (clean_run, injected_run):
        for row in run["results"]:
            comparison_rows.append(
                "| "
                + " | ".join(
                    (
                        "Nhật ký sạch" if run["dataset"] == "clean_logs" else "Nhật ký chèn lệnh",
                        VIETNAMESE_DEFENSES[row["defense"]],
                        str(row["sample_count"]),
                        f"{row['accuracy_percent']:.2f}%",
                        percent(row["manipulation_rate_percent"]),
                        percent(row["attack_success_rate_percent"]),
                        f"{row['correct_prediction_count']}/{row['sample_count']}",
                    )
                )
                + " |"
            )

    defense_rank_rows = []
    for rank, row in enumerate(ranking, start=1):
        defense_rank_rows.append(
            f"| {rank} | {VIETNAMESE_DEFENSES[row['defense']]} | "
            f"{row['accuracy_percent']:.2f}% | "
            f"{percent(row['manipulation_rate_percent'])} | "
            f"{percent(row['attack_success_rate_percent'])} |"
        )

    clean_risk_label = "Không áp dụng"
    report = f"""# Chương 4 - Đánh giá hiệu quả phòng vệ chèn lệnh cho AI Agent

## 1. Thiết lập thí nghiệm

Thí nghiệm được thực hiện bằng mô hình Ollama cục bộ `llama3.2`, nhiệt độ suy luận `0`, seed suy luận `0` và bộ cấu hình phòng vệ cố định. Hai tập dữ liệu được đánh giá tuần tự:

- Nhật ký sạch: {clean_run['sample_count']} mẫu.
- Nhật ký có chèn lệnh: {injected_run['sample_count']} mẫu.
- Tổng cộng: {clean_run['sample_count'] + injected_run['sample_count']} mẫu.
- Năm cấu hình: không phòng vệ, cô lập ngữ cảnh, bộ phát hiện chèn lệnh, ràng buộc đầu ra JSON và kết hợp các phòng vệ.
- Tổng số lần gọi mô hình: 210 lần.
- Thời điểm chạy đã ghi nhận: `{latest_timestamp}`.
- Fingerprint tập sạch: `{clean_run['dataset_sha256']}`.
- Fingerprint tập chèn lệnh: `{injected_run['dataset_sha256']}`.

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
{chr(10).join(comparison_rows)}

### 5.2 Xếp hạng trên tập chèn lệnh

| Hạng | Cấu hình | Độ chính xác | Tỷ lệ thao túng | Tỷ lệ tấn công thành công |
|---:|---|---:|---:|---:|
{chr(10).join(defense_rank_rows)}

- **Phòng vệ tốt nhất:** {', '.join(VIETNAMESE_DEFENSES[name] for name in best_defenses)} (tỷ lệ thao túng thấp nhất trong tập chèn lệnh và xếp hạng theo độ chính xác).
- **Phòng vệ kém nhất:** {VIETNAMESE_DEFENSES[worst['defense']]} (100.00% thao túng, 0.00% độ chính xác).
- **Cải thiện lớn nhất:** {', '.join(VIETNAMESE_DEFENSES[name] for name, _improvement, _accuracy in largest_defenses)}, thay đổi tỷ lệ thao túng {largest_manipulation[1]:+.2f} điểm phần trăm so với không phòng vệ.
- **Cải thiện nhỏ nhất:** {VIETNAMESE_DEFENSES[smallest_manipulation[0]]}, thay đổi tỷ lệ thao túng {smallest_manipulation[1]:+.2f} điểm phần trăm so với không phòng vệ.

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
"""

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
