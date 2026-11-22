def research_dashboard() -> str:
    return """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Đánh giá bảo mật AI Agent</title>
  <style>
    :root { color-scheme: light; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #172033; background: #f3f5f8; }
    * { box-sizing: border-box; }
    body { max-width: 1180px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
    header { margin-bottom: 1.4rem; }
    h1 { margin: 0; font-size: clamp(1.7rem, 3vw, 2.25rem); letter-spacing: -.03em; }
    h2 { margin: 0 0 .8rem; font-size: 1.08rem; }
    p { margin: .35rem 0; }
    a { color: #174ea6; }
    .eyebrow { color: #526179; font-size: .75rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    .muted { color: #64748b; }
    .card { margin: 1rem 0; padding: 1.2rem; border: 1px solid #dce2eb; border-radius: 12px; background: #fff; box-shadow: 0 2px 6px rgb(21 34 56 / 4%); }
    .toolbar { display: flex; flex-wrap: wrap; align-items: end; gap: 1rem; }
    .field { display: grid; flex: 1 1 200px; gap: .35rem; font-weight: 650; }
    select, input[type=text] { min-height: 42px; padding: .5rem .65rem; border: 1px solid #cbd4e1; border-radius: 7px; background: #fff; color: inherit; font: inherit; }
    button, .export-link { min-height: 40px; padding: .55rem .85rem; border: 0; border-radius: 7px; color: #fff; background: #174ea6; font: inherit; font-weight: 650; cursor: pointer; text-decoration: none; }
    button:disabled { opacity: .6; cursor: wait; }
    .defenses { display: flex; flex-wrap: wrap; gap: .6rem 1rem; margin-top: 1rem; }
    .defenses label { display: flex; align-items: center; gap: .35rem; }
    .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; }
    .metric { padding: 1rem; border: 1px solid #dce2eb; border-radius: 10px; background: #fff; }
    .metric strong { display: block; font-size: 1.55rem; }
    .metric span { color: #526179; font-size: .86rem; }
    .chart-row { display: grid; grid-template-columns: minmax(140px, 1fr) minmax(180px, 3fr) 4rem; align-items: center; gap: .8rem; margin: .75rem 0; }
    progress { width: 100%; height: 1rem; accent-color: #174ea6; }
    .rate-bad progress { accent-color: #c62828; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    th, td { min-width: 110px; padding: .7rem; border-bottom: 1px solid #e5e9f0; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
    th { color: #526179; font-size: .76rem; text-transform: uppercase; }
    td.detail { min-width: 240px; white-space: pre-wrap; }
    .pass { color: #11632e; font-weight: 700; }
    .fail { color: #9c2020; font-weight: 700; }
    .exports { display: flex; flex-wrap: wrap; gap: .65rem; }
    .exports a[aria-disabled=true] { opacity: .45; pointer-events: none; }
    .chart-images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
    .chart-images[hidden] { display: none; }
    .chart-images figure { margin: 0; min-width: 0; }
    .chart-images img { display: block; width: 100%; height: auto; border: 1px solid #e5e9f0; border-radius: 8px; }
    .chart-images figcaption { margin-top: .35rem; color: #526179; font-weight: 650; }
    #status { min-height: 1.5rem; color: #526179; }
    #status.error { color: #a32020; }
    @media (max-width: 700px) { body { padding: 1.2rem .8rem 2rem; } .metric-grid, .chart-images { grid-template-columns: 1fr; } .chart-row { grid-template-columns: 1fr 2fr 3rem; } }
    @media (max-width: 420px) { .metric-grid { grid-template-columns: 1fr 1fr; } .chart-row { grid-template-columns: 1fr 2rem; } .chart-row progress { grid-column: 1 / -1; grid-row: 2; } }
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">Nghiên cứu an toàn thông tin</p>
    <h1>Đánh giá bảo mật AI Agent</h1>
    <p class="muted">Đo lường mức độ ảnh hưởng của chỉ dẫn độc hại được nhúng trong nhật ký và so sánh các cơ chế phòng vệ.</p>
    <p><a href="/ui">← Quay lại bảng điều khiển phân tích nhật ký</a></p>
  </header>

  <section class="card" aria-labelledby="run-heading">
    <h2 id="run-heading">Chạy thực nghiệm</h2>
    <p class="muted">Mỗi cấu hình chạy riêng trên cùng tập dữ liệu. Mô hình Ollama cục bộ được gọi tuần tự; khi chọn tất cả, thời gian chạy phụ thuộc vào số mẫu và phần cứng.</p>
    <div class="toolbar">
      <label class="field" for="dataset">Tập dữ liệu
        <select id="dataset">
          <option value="combined">Kết hợp (sạch + chèn lệnh)</option>
          <option value="clean_logs">Nhật ký sạch (đối chứng)</option>
          <option value="injected_logs">Nhật ký có chèn lệnh</option>
        </select>
      </label>
      <label class="field" for="experiment-name">Tên thực nghiệm
        <input id="experiment-name" type="text" value="Prompt injection defense benchmark" maxlength="120">
      </label>
      <button id="run-button" type="button">Chạy đánh giá</button>
      <button id="run-all-button" type="button">Chạy tất cả thực nghiệm</button>
    </div>
    <fieldset style="border:0;padding:0;margin:1rem 0 0">
      <legend><strong>Cấu hình phòng vệ</strong></legend>
      <div class="defenses">
        <label><input type="checkbox" name="defense" value="No Defense" checked> Không phòng vệ</label>
        <label><input type="checkbox" name="defense" value="Context Isolation" checked> Cô lập ngữ cảnh</label>
        <label><input type="checkbox" name="defense" value="Prompt Injection Detector" checked> Bộ phát hiện chèn lệnh</label>
        <label><input type="checkbox" name="defense" value="Structured Output Constraint" checked> Ràng buộc đầu ra JSON</label>
        <label><input type="checkbox" name="defense" value="Combined Defenses" checked> Kết hợp các phòng vệ</label>
      </div>
    </fieldset>
    <p id="status" role="status" aria-live="polite"></p>
  </section>

  <section class="metric-grid" aria-label="Chỉ số thực nghiệm">
    <div class="metric"><strong id="dataset-size">—</strong><span>Số mẫu trong lần chạy</span></div>
    <div class="metric"><strong id="accuracy">—</strong><span>Độ chính xác</span></div>
    <div class="metric"><strong id="manipulation-rate">—</strong><span>Tỷ lệ bị thao túng</span></div>
    <div class="metric"><strong id="attack-success-rate">—</strong><span>Tỷ lệ tấn công thành công</span></div>
  </section>

  <section class="card">
    <h2>So sánh hiệu quả phòng vệ</h2>
    <div id="charts"><p class="muted">Chạy thực nghiệm để hiển thị biểu đồ.</p></div>
    <div id="saved-charts" class="chart-images" hidden>
      <figure><img id="accuracy-chart" alt="Biểu đồ so sánh độ chính xác"><figcaption>So sánh độ chính xác (tập nhật ký có chèn lệnh)</figcaption></figure>
      <figure><img id="manipulation-chart" alt="Biểu đồ so sánh tỷ lệ bị thao túng"><figcaption>So sánh tỷ lệ bị thao túng</figcaption></figure>
    </div>
  </section>

  <section class="card">
    <h2>Kết quả theo cấu hình</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Cấu hình</th><th>Độ chính xác</th><th>Tỷ lệ bị thao túng</th><th>Tấn công bị thành công</th><th>Dự đoán đúng</th></tr></thead>
        <tbody id="comparison-body"><tr><td colspan="5" class="muted">Chưa có kết quả thực nghiệm.</td></tr></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>Chi tiết mẫu bị thao túng hoặc phân loại sai</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Mã mẫu</th><th>Loại</th><th>Nhãn thật</th><th>Dự đoán</th><th>Phát hiện chèn lệnh</th><th>Kết quả</th><th>Lý do mô hình</th></tr></thead>
        <tbody id="details-body"><tr><td colspan="7" class="muted">Chưa có kết quả thực nghiệm.</td></tr></tbody>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>Xuất dữ liệu và hỗ trợ luận văn</h2>
    <p class="muted">Các tệp được tạo sau lần chạy thành công. Bản tóm tắt Markdown nêu rõ định nghĩa phép đo và giới hạn của kết quả.</p>
    <div class="exports">
      <a class="export-link" data-export href="/api/research/export/results.json" aria-disabled="true">Tải results.json</a>
      <a class="export-link" data-export href="/api/research/export/results.csv" aria-disabled="true">Tải results.csv</a>
      <a class="export-link" data-export href="/api/research/export/report.md" aria-disabled="true">Tải research_report.md</a>
      <a class="export-link" data-export-chart href="/api/research/export/accuracy_comparison.png" aria-disabled="true">Tải biểu đồ độ chính xác</a>
      <a class="export-link" data-export-chart href="/api/research/export/manipulation_rate_comparison.png" aria-disabled="true">Tải biểu đồ tỷ lệ thao túng</a>
      <a class="export-link" data-export-chart href="/api/research/export/defense_effectiveness.png" aria-disabled="true">Tải biểu đồ hiệu quả phòng vệ</a>
    </div>
    <p id="history-info" class="muted">Đang tải thông tin tập dữ liệu…</p>
  </section>

  <script>
    const status = document.getElementById("status");
    const runButton = document.getElementById("run-button");
    const runAllButton = document.getElementById("run-all-button");
    const comparisonBody = document.getElementById("comparison-body");
    const detailsBody = document.getElementById("details-body");
    const datasetSelector = document.getElementById("dataset");
    let datasetCounts = null;

    function addCell(row, value, className = "") {
      const cell = document.createElement("td");
      cell.textContent = value ?? "—";
      if (className) cell.className = className;
      row.append(cell);
    }

    function percent(value) {
      return value === null || value === undefined ? "N/A" : `${value.toFixed(2)}%`;
    }

    function renderExperiment(experiment) {
      document.getElementById("dataset-size").textContent =
        `${experiment.sample_count} (${experiment.injected_sample_count} chèn lệnh)`;
      const best = experiment.results.reduce((winner, result) => {
        if (!winner) return result;
        const winnerRate = winner.manipulation_rate_percent ?? Infinity;
        const resultRate = result.manipulation_rate_percent ?? Infinity;
        return resultRate < winnerRate ||
          (resultRate === winnerRate && result.accuracy_percent > winner.accuracy_percent)
          ? result : winner;
      }, null);
      document.getElementById("accuracy").textContent = best ? percent(best.accuracy_percent) : "—";
      document.getElementById("manipulation-rate").textContent =
        best ? percent(best.manipulation_rate_percent) : "—";
      document.getElementById("attack-success-rate").textContent =
        best ? percent(best.attack_success_rate_percent) : "—";

      comparisonBody.replaceChildren();
      const chartContainer = document.getElementById("charts");
      chartContainer.replaceChildren();
      for (const result of experiment.results) {
        const row = document.createElement("tr");
        addCell(row, result.defense);
        addCell(row, percent(result.accuracy_percent));
        addCell(row, percent(result.manipulation_rate_percent));
        addCell(row, percent(result.attack_success_rate_percent));
        addCell(row, `${result.correct_prediction_count}/${result.sample_count}`);
        comparisonBody.append(row);

        const chartRow = document.createElement("div");
        chartRow.className = "chart-row";
        const name = document.createElement("span");
        name.textContent = result.defense;
        const accuracy = document.createElement("progress");
        accuracy.max = 100;
        accuracy.value = result.accuracy_percent;
        accuracy.setAttribute("aria-label", `Độ chính xác ${result.defense}`);
        const accuracyLabel = document.createElement("strong");
        accuracyLabel.textContent = percent(result.accuracy_percent);
        chartRow.append(name, accuracy, accuracyLabel);
        chartContainer.append(chartRow);

        if (result.manipulation_rate_percent !== null) {
          const attackRow = document.createElement("div");
          attackRow.className = "chart-row rate-bad";
          const attackName = document.createElement("span");
          attackName.textContent = `Bị thao túng · ${result.defense}`;
          const attackProgress = document.createElement("progress");
          attackProgress.max = 100;
          attackProgress.value = result.manipulation_rate_percent;
          attackProgress.setAttribute("aria-label", `Tỷ lệ bị thao túng ${result.defense}`);
          const attackLabel = document.createElement("strong");
          attackLabel.textContent = percent(result.manipulation_rate_percent);
          attackRow.append(attackName, attackProgress, attackLabel);
          chartContainer.append(attackRow);
        }
      }

      const selected = experiment.results.find((result) => result.defense === best?.defense);
      detailsBody.replaceChildren();
      const failures = (selected?.predictions ?? []).filter(
        (prediction) => prediction.manipulated || !prediction.correct,
      );
      if (!failures.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 7;
        cell.textContent = "Không có mẫu sai hoặc bị thao túng trong cấu hình tốt nhất.";
        cell.className = "pass";
        row.append(cell);
        detailsBody.append(row);
      } else {
        for (const prediction of failures) {
          const row = document.createElement("tr");
          addCell(row, prediction.sample_id);
          addCell(row, prediction.is_injected ? "Có chèn lệnh" : "Sạch");
          addCell(row, `${prediction.ground_truth_risk} · ${prediction.ground_truth_attack_type}`);
          addCell(row, `${prediction.predicted_risk} · ${prediction.predicted_attack_type}`);
          addCell(row, prediction.injection_detected ? "Có" : "Không");
          addCell(row, prediction.manipulated ? "Bị thao túng" : "Phân loại sai",
            prediction.manipulated ? "fail" : "");
          addCell(row, prediction.reason, "detail");
          detailsBody.append(row);
        }
      }
      for (const link of document.querySelectorAll("[data-export]")) {
        link.setAttribute("aria-disabled", "false");
      }
      const timestamp = new Date(experiment.timestamp).toLocaleString();
      document.getElementById("history-info").textContent =
        `Thực nghiệm: ${experiment.experiment_name} · Mô hình: ${experiment.model} · ${timestamp}`;
    }

    function renderFullBenchmark(batch) {
      const cleanRun = batch.runs.find((run) => run.dataset === "clean_logs");
      const injectedRun = batch.runs.find((run) => run.dataset === "injected_logs");
      if (!cleanRun || !injectedRun) {
        throw new Error("Kết quả thực nghiệm chưa có đủ hai tập dữ liệu.");
      }

      const best = injectedRun.results.find((result) => result.defense === batch.best_defense);
      document.getElementById("dataset-size").textContent =
        `${cleanRun.sample_count + injectedRun.sample_count} ` +
        `(${injectedRun.sample_count} chèn lệnh)`;
      document.getElementById("accuracy").textContent =
        best ? percent(best.accuracy_percent) : "—";
      document.getElementById("manipulation-rate").textContent =
        best ? percent(best.manipulation_rate_percent) : "—";
      document.getElementById("attack-success-rate").textContent =
        best ? percent(best.attack_success_rate_percent) : "—";

      comparisonBody.replaceChildren();
      for (const experiment of batch.runs) {
        for (const result of experiment.results) {
          const row = document.createElement("tr");
          addCell(row, `${experiment.dataset} · ${result.defense}`);
          addCell(row, percent(result.accuracy_percent));
          addCell(row, percent(result.manipulation_rate_percent));
          addCell(row, percent(result.attack_success_rate_percent));
          addCell(row, `${result.correct_prediction_count}/${result.sample_count}`);
          comparisonBody.append(row);
        }
      }

      const chartContainer = document.getElementById("charts");
      chartContainer.replaceChildren();
      const bestWorst = document.createElement("p");
      bestWorst.textContent =
        `Tốt nhất trên tập chèn lệnh: ${batch.best_defense} · ` +
        `Kém nhất: ${batch.worst_defense}.`;
      chartContainer.append(bestWorst);

      const detailsRun = injectedRun.results.find(
        (result) => result.defense === batch.best_defense,
      );
      detailsBody.replaceChildren();
      const failures = (detailsRun?.predictions ?? []).filter(
        (prediction) => prediction.manipulated || !prediction.correct,
      );
      if (!failures.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 7;
        cell.textContent = "Không có mẫu sai hoặc bị thao túng trong cấu hình tốt nhất.";
        cell.className = "pass";
        row.append(cell);
        detailsBody.append(row);
      } else {
        for (const prediction of failures) {
          const row = document.createElement("tr");
          addCell(row, prediction.sample_id);
          addCell(row, "Có chèn lệnh");
          addCell(row, `${prediction.ground_truth_risk} · ${prediction.ground_truth_attack_type}`);
          addCell(row, `${prediction.predicted_risk} · ${prediction.predicted_attack_type}`);
          addCell(row, prediction.injection_detected ? "Có" : "Không");
          addCell(row, prediction.manipulated ? "Bị thao túng" : "Phân loại sai",
            prediction.manipulated ? "fail" : "");
          addCell(row, prediction.reason, "detail");
          detailsBody.append(row);
        }
      }

      const chartVersion = encodeURIComponent(batch.timestamp);
      document.getElementById("accuracy-chart").src =
        `/api/research/export/accuracy_comparison.png?v=${chartVersion}`;
      document.getElementById("manipulation-chart").src =
        `/api/research/export/manipulation_rate_comparison.png?v=${chartVersion}`;
      document.getElementById("saved-charts").hidden = false;
      for (const link of document.querySelectorAll("[data-export], [data-export-chart]")) {
        link.setAttribute("aria-disabled", "false");
      }
      document.getElementById("history-info").textContent =
        `Thực nghiệm: ${batch.experiment_name} · Mô hình: ${batch.model} · ` +
        `Seed: ${batch.seed} · Tốt nhất: ${batch.best_defense} · ` +
        `Kém nhất: ${batch.worst_defense}. ` +
        new Date(batch.timestamp).toLocaleString();
    }

    async function loadHistory() {
      const response = await fetch("/api/research/results");
      if (!response.ok) throw new Error("Không thể tải lịch sử thực nghiệm.");
      const data = await response.json();
      datasetCounts = data.dataset_counts;
      updateDatasetSize();
      if (data.latest_experiment) renderExperiment(data.latest_experiment);
      document.getElementById("history-info").textContent =
        `Tập sạch: ${data.dataset_counts.clean_logs} mẫu · ` +
        `Tập chèn lệnh: ${data.dataset_counts.injected_logs} mẫu · ` +
        `${data.history.experiments.length} lần thực nghiệm đã lưu.`;
      if (data.exports_available) {
        for (const link of document.querySelectorAll("[data-export]")) {
          link.setAttribute("aria-disabled", "false");
        }
      }
      if (data.charts_available) {
        for (const link of document.querySelectorAll("[data-export-chart]")) {
          link.setAttribute("aria-disabled", "false");
        }
        const latestRun = data.latest_experiment;
        if (latestRun?.dataset === "injected_logs") {
          const cacheKey = encodeURIComponent(latestRun.timestamp);
          document.getElementById("accuracy-chart").src =
            `/api/research/export/accuracy_comparison.png?v=${cacheKey}`;
          document.getElementById("manipulation-chart").src =
            `/api/research/export/manipulation_rate_comparison.png?v=${cacheKey}`;
          document.getElementById("saved-charts").hidden = false;
        }
      }
    }

    function updateDatasetSize() {
      if (!datasetCounts) return;
      const selected = datasetSelector.value;
      const count = selected === "combined"
        ? datasetCounts.clean_logs + datasetCounts.injected_logs
        : datasetCounts[selected];
      const injected = selected === "clean_logs" ? 0 : datasetCounts.injected_logs;
      document.getElementById("dataset-size").textContent =
        `${count} (${injected} chèn lệnh)`;
    }

    datasetSelector.addEventListener("change", updateDatasetSize);

    runButton.addEventListener("click", async () => {
      const defenses = [...document.querySelectorAll('input[name="defense"]:checked')]
        .map((input) => input.value);
      if (!defenses.length) {
        status.className = "error";
        status.textContent = "Hãy chọn ít nhất một cấu hình phòng vệ.";
        return;
      }
      runButton.disabled = true;
      runAllButton.disabled = true;
      status.className = "";
      status.textContent = "Đang chạy thực nghiệm bằng mô hình Ollama cục bộ…";
      try {
        const response = await fetch("/api/research/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset: datasetSelector.value,
            defenses,
            experiment_name: document.getElementById("experiment-name").value,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Không thể chạy thực nghiệm.");
        renderExperiment(data);
        status.textContent = "Thực nghiệm hoàn tất và các kết quả đã được lưu.";
      } catch (error) {
        status.className = "error";
        status.textContent = error.message || "Không thể chạy thực nghiệm.";
      } finally {
        runButton.disabled = false;
        runAllButton.disabled = false;
      }
    });

    runAllButton.addEventListener("click", async () => {
      runAllButton.disabled = true;
      runButton.disabled = true;
      status.className = "";
      status.textContent =
        "Đang chạy 10 cấu hình trên hai tập dữ liệu (210 lượt gọi mô hình). " +
        "Quá trình chạy tuần tự và có thể mất vài phút…";
      try {
        const response = await fetch("/api/research/run-all", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            experiment_name: document.getElementById("experiment-name").value,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Không thể hoàn tất toàn bộ thực nghiệm.");
        renderFullBenchmark(data);
        status.textContent =
          `Đã hoàn thành toàn bộ thực nghiệm. Đã lưu ${data.runs.length} tập kết quả, ` +
          "CSV, báo cáo Markdown và hai biểu đồ PNG.";
      } catch (error) {
        status.className = "error";
        status.textContent = error.message || "Không thể hoàn tất toàn bộ thực nghiệm.";
      } finally {
        runAllButton.disabled = false;
        runButton.disabled = false;
      }
    });

    loadHistory().catch((error) => {
      status.className = "error";
      status.textContent = error.message;
    });
  </script>
</body>
</html>"""
