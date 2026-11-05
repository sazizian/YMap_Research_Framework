Y-MAP RUN AUDIT — 23 OCTOBER 2026

START HERE
Open Run_Audit_Report.html. The findings directory contains machine-readable
results. No profiling input, cost label, prediction, or original framework file
has been changed. This package does not contain the user's original 28 MB input
ZIP; keep that ZIP with the experiment.

WHAT WAS REVIEWED
728 current generated schemas, 11 source systems, PostgreSQL measurement logs,
and evaluation_20261023_105939_243. The protocol uses generated logical records
and actual database timings. It is a small new protocol, not the manuscript's
original dataset or full pretrained multi-view experiment.

NEXT: CHECK THE TRAINING MACHINE
Put this ZIP in the YMap_Research_Framework project directory. From its PowerShell
terminal, extract it without overwriting existing audit files:

Expand-Archive -LiteralPath .\YMap_Run_Audit_20261023.zip -DestinationPath .
.\.venv\Scripts\python.exe .\YMap_Run_Audit_20261023\scripts\collect_neural_readiness.py --project-root .

Share the printed readiness JSON. This checks installed packages, NVIDIA device
information when available, the configured device, and local checkpoint-directory
presence. It does not install software, download weights, train a model, contact
a database, or print database credentials. A missing nvidia-smi command alone does
not establish that the computer has no GPU. Checkpoint directory presence does
not establish model completeness or compatibility.

EXISTING-RESULT INTERPRETATION
- Consistent split/metric records support software execution, not independent
  authentication of a database session or confirmation of manuscript results.
- The 16-feature baseline loses distinctions between many physical designs.
- Timing variability also affects the measured exact Pareto reference.
- No tested recommendation comparison is significant after the declared Holm
  adjustment. This does not prove that the methods are equivalent.
- Existing preference_regret is best-in-returned-set regret. The derived report
  adds actual predicted-top-one regret, with both normalizations clearly named.
- Hypervolume and normalized error are not accuracy percentages.

OPTIONAL: REPRODUCE THE READ-ONLY AUDITS
Extract the ORIGINAL evaluation_review_20261023_113149_551.zip to a new directory,
for example C:\YMap\review_input. That directory must contain candidates.jsonl,
measurements/, profile.yaml and the evaluation directory. The commands below
assume that the current directory is your original YMap project root. Use fresh
output directories each time.

.\.venv\Scripts\python.exe .\YMap_Run_Audit_20261023\scripts\audit_evaluation.py --input C:\YMap\review_input --out .\runs\audit_consistency_01
.\.venv\Scripts\python.exe .\YMap_Run_Audit_20261023\scripts\audit_measurements.py C:\YMap\review_input --output .\runs\audit_measurements_01
.\.venv\Scripts\python.exe .\YMap_Run_Audit_20261023\scripts\audit_representation.py --input C:\YMap\review_input --output .\runs\audit_representation_01 --framework-source .\src
.\.venv\Scripts\python.exe .\YMap_Run_Audit_20261023\scripts\recompute_preference_regret.py --review-root C:\YMap\review_input --out .\runs\audit_regret_01

The numerical consistency audit requires NumPy, pandas and SciPy, already used
by the framework. The representation audit calls the project's canonical schema
and logical-oracle helpers; it does not execute SQL. It must be used with the
same framework version as the reviewed run. The remaining audit scripts use
Python's standard library. The consistency script intentionally expects this
728-schema/11-system release and is not a general benchmark validator.

OPTIONAL: HAND-CALCULATED REGRET TESTS
.\.venv\Scripts\python.exe -m unittest discover -s .\YMap_Run_Audit_20261023\scripts -p test_recompute_preference_regret.py -v

FILES
Run_Audit_Report.html: readable report.
findings/: JSON/CSV results from the independent audit routines.
scripts/: reproducible audits and local readiness check, all in English.
VALIDATION.json: verification scope and checks performed for this package.
MANIFEST.json: SHA-256 and byte length for each other packaged file.

SCIENTIFIC NEXT STEP
Preserve this run as a baseline. A graph-plus-symbolic ablation can reuse these
measurements and test a richer representation, but is not the full three-text-
encoder model. Full multi-view modeling needs the actual authorized checkpoints.
Do not promise better scores in advance. Any revisions suggested by inspecting
these outer-test results are exploratory; confirmatory claims require an honest
nested or independent reassessment. Do not launch a full new database profiling
run solely to improve a score.
