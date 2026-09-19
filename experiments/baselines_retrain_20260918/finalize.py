"""Freeze both finished models' predictions, then apply the unchanged real-test scorer."""
import argparse,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];proto=json.loads((RUN/'protocol.json').read_text());OLD=Path(proto['test_gold']);OUT=RUN/'test_results'
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','freeze','score','all'],default='all');p.add_argument('--model');args=p.parse_args()
if args.stage in ['prepare','all']:
 OUT.mkdir(exist_ok=True)
 for name in ['dataset_manifest.json','label_audit.json']:shutil.copy2(OLD/name,OUT/name)
 if not (OUT/'bundles').exists():(OUT/'bundles').symlink_to(OLD/'bundles',target_is_directory=True)
 selected={m:json.loads((RUN/'training'/m/'best.json').read_text()) for m in ['polytune','laddersym']}
 (OUT/'checkpoint_selection.json').write_text(json.dumps({'models':selected},indent=2))
 (OUT/'inference_launch.json').write_text(json.dumps({'jobs':{m:{'predictions':str(RUN/'training'/m/'test_predictions')} for m in selected}},indent=2))
if args.stage=='all':
 for m in ['polytune','laddersym']:subprocess.run([sys.executable,__file__,'--stage','freeze','--model',m],check=True)
 subprocess.run([sys.executable,__file__,'--stage','score'],check=True)
elif args.stage in ['freeze','score']:
 spec=importlib.util.spec_from_file_location('frozen_real_scorer',OLD/'evaluate.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.HERE=OUT
 if args.stage=='freeze':mod.freeze(args.model)
 else:
  mod.score();r=json.loads((OUT/'results.json').read_text())
  lines=['# 重训后的真实测试结果','','测试集：034.zip，036–040 使用替换包；标签、转换规则和计分规则与修正版旧评测一致。模型由合成验证集生成 F1 选择。','','| 集合 | 模型 | Precision | Recall | F1@50ms | F1@IoU≥0.3 |','|---|---|---:|---:|---:|---:|']
  for subset in ['full40','filtered30']:
   for model in ['polytune','laddersym']:
    x=r['models'][model]['subsets'][subset]['results']['five_type'];v=x['onset_50ms']['micro'];iou=x['iou_0.3']['micro']['f1'];lines.append(f"| {subset} | {model} | {v['precision']:.6f} | {v['recall']:.6f} | {v['f1']:.6f} | {iou:.6f} |")
  lines+=['','这些是时间匹配错误事件诊断指标，不是论文的 canonical combined-pipeline F1。完整分类、置信区间及逐录音结果见 results.json。']
  (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
