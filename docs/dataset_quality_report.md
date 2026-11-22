# Dataset Quality Report

## 1. Scope and Purpose

This report reviews the current clean and prompt-injected log datasets used for the AI Agent security benchmark. It measures sample size, risk-label balance, attack-type coverage, injection-category diversity, and duplicate-ID risk.

## 2. Dataset Sizes

- Clean logs: **22** samples
- Injected logs: **20** samples
- Combined benchmark: **42** samples

## 3. Clean Dataset Distribution

### Risk levels

| Risk Level | Count |
| --- | --- |
| Low | 7 |
| Medium | 9 |
| High | 6 |

### Attack types

| Attack Type | Count |
| --- | --- |
| None | 7 |
| Brute Force | 3 |
| SQL Injection | 3 |
| Port Scan | 3 |
| Malware | 3 |
| Command and Control | 3 |

## 4. Injected Dataset Distribution

### Risk levels

| Risk Level | Count |
| --- | --- |
| Low | 0 |
| Medium | 8 |
| High | 12 |

### Attack types

| Attack Type | Count |
| --- | --- |
| Brute Force | 4 |
| SQL Injection | 4 |
| Port Scan | 4 |
| Malware | 4 |
| Command and Control | 4 |

### Injection categories

| Injection Category | Count |
| --- | --- |
| Direct Prompt Injection | 10 |
| Indirect Prompt Injection | 5 |
| Context Manipulation | 5 |

## 5. Quality Assessment

- The clean dataset is at the upper end of the starter target at 22 samples; its five security attack classes each have three clean examples.
- The injected dataset now has 20 samples, with four examples for each of the five attack types.
- Direct, indirect, and context-manipulation injections are represented as separate, reusable dataset metadata.
- A remaining weakness is that most injected samples retain High risk ground truth, so low-risk and medium-risk injected edge cases should be expanded further in future rounds.
- Clean attack types absent from the corpus: None.
- Injection categories absent from the corpus: None.
- Duplicate ID count across the corpus: 0.

## 6. Weaknesses and Recommended Next Data

1. Add paired examples with Medium risk in the injected corpus rather than concentrating ground-truth risk in High.
2. Add low-risk baseline traffic from several protocols, including DNS, SMTP, database, remote-access, and internal API traffic.
3. Add multi-line and mixed-signal logs so prompt-injection text cannot be separated by a unique synthetic marker.
4. Add additional indirect injections such as ticket comments, alert notes, approved-change context, and analyst summaries.
5. Preserve the JSON schema and paired clean/injected identifiers when adding samples, then rerun dataset-quality validation and the complete benchmark.

## 7. Reproducibility

Run `python tools/analyze_dataset_quality.py` to regenerate this report from the JSON dataset files. The report is descriptive and should be rerun after each corpus revision.
