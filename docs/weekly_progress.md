# Weekly Research Progress

## Study Week

- Expanded the clean dataset from 7 to 22 samples.
- Expanded the injected dataset from 5 to 20 samples.
- Added direct, indirect, and context-manipulation prompt-injection categories.
- Balanced the clean attack-type coverage across Brute Force, SQL Injection,
  Port Scan, Malware, and Command and Control.
- Prepared the benchmark to run 210 model calls for a complete clean/injected
  run across five defense configurations.
- Added defense-effectiveness PNG output alongside accuracy and manipulation
  comparison charts.

## Validation Status

- Expanded suite passes with the larger corpus.
- Corrected real benchmark completed with `llama3.2`, seed 0 and 210 model calls.
- Clean accuracy ranged from 50.00% to 59.09%.
- Injected manipulation rate ranged from 95.00% to 100.00%.
- Generated JSON, CSV, Markdown and PNG artifacts under `data/research/`.
- Generated artifacts remain excluded from Git.

## Next Actions

1. Preserve the corrected benchmark outputs as research evidence.
2. Expand ground-truth Medium/Low coverage in injected logs.
3. Run repeated seeds and additional local models.
4. Analyze manipulation outcomes by injection category and attack type.
