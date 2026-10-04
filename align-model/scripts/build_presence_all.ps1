$env:PYTHONPATH = "src;scripts;..\DataCreate\src;..\synth-pipeline\src"
$py = ".venv-amt-bench\Scripts\python.exe"
$jobs = @(
  @("runs\realistic92-stack-v2\cache-dual-r3", "runs\realistic92-aligner-v1\ALIGNER_FREEZE.json", "r92"),
  @("runs\fast102-v1\cache-dual-r3", "runs\fast102-v1\ALIGNER_FREEZE.json", "fast"),
  @("runs\dclike11-v1\cache-dual-r3", "runs\dclike11-v1\ALIGNER_FREEZE.json", "d11")
)
foreach ($j in $jobs) {
  foreach ($split in @("train", "val")) {
    & $py scripts\build_presence_dataset.py --cache $j[0] --freeze $j[1] --split $split --workers 5 --output "runs\precision-v4\presence\$($j[2])-$split.npz"
  }
}
