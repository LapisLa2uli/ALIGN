"""Package read-only findings and playable excerpts for human adjudication."""
from pathlib import Path
from collections import Counter
import base64
import csv
import hashlib
import html
import io
import json

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
structure = json.loads((HERE / "structure.json").read_text())
checks = {x["key"]: x for x in json.loads((HERE / "pyin_checks.json").read_text())}

cases = [
    dict(sample="002", start=67.6397, end=67.660091, clip_start=64.8, clip_end=67.660091,
         title="确定的重复记录：4 条完全相同的 missed_note",
         finding="四条标签的时间、谱面位置、音高列表等内容完全一致，仅 ID 不同。它们位于录音最后约 20 ms；这是确定的重复计数问题，不能当四个不同漏音。另有一条 61.853–62.208 秒的漏音指向相同谱面位置，需要一起复核。"),
    dict(sample="005", start=13.5414, end=14.9662, clip_start=12.8, clip_end=15.8,
         title="同类标注重复覆盖，且谱面字段不一致",
         finding="13.5414–14.9662 秒的整段 wrong_note，包含 13.5414–13.8932 与 14.7751–14.9662 两个单独 wrong_note。需要明确采用整段还是逐音符计数，避免重复。14.7751 秒的核心 UI ID 指向第 44 个发声音符，core range 却指向第 45 个。",
         pyin_key="005_label4"),
    dict(sample="036", start=5.2076, end=5.6153, clip_start=4.7, clip_end=6.2,
         title="新增三处错音：后两段缺乏稳定音高，需检查时间边界",
         finding="5.3435–5.4794 与 5.4794–5.6153 秒的 RMS 分别约 0.0022/0.0013，pYIN 没有可靠有声音高帧，更接近低能量/停顿区域。不能据此直接判定无错误，但当前证据不足以支持两个独立 wrong_note；先核对标注是否移到了停顿。前一段 5.2076–5.3435 秒有音高，仍需核对谱面对应位置。",
         pyin_key="036_label0"),
    dict(sample="039", start=2.4552, end=2.5671, clip_start=1.9, clip_end=3.05,
         title="较强的错标/落点疑点：音高与所选谱面一致",
         finding="所选谱面 F5（MIDI 77，B♭ 单簧管实际发声 E♭5/MIDI 75）；录音 pYIN 中位为 MIDI 75.1，即约 +10 音分。YIN 也一致，附近序列能对上。因此该段本身没有明确错音证据，应检查是否想标相邻的音，而时间或选区偏了一格。",
         pyin_key="039_label0", expected=75),
    dict(sample="039", start=17.4765, end=17.6419, clip_start=16.9, clip_end=18.2,
         title="有音高不一致证据，但仍需确认谱面位置",
         finding="所选核心音符对应实际发声 MIDI 79，录音该段主要约 MIDI 75.1。音高不一致有声学依据；但整段存在序列对齐歧义，不能仅凭此确认它一定是当前选中音符的 wrong_note。",
         pyin_key="039_label1", expected=79),
    dict(sample="031", start=25.73, end=25.85, clip_start=24.85, clip_end=26.65,
         title="空标注的音准复核候选，不是已确认的 wrong_note 漏标",
         finding="邻近谱面序列能对应，目标应为实际发声 F4/MIDI 65；YIN/pYIN 估计约 65.6，频谱主峰约 362 Hz，偏高约 60 音分。初筛四舍五入造成“高一个半音”的表象，精查后已撤回该判断。周围音符也高约 13–36 音分，应由标注者判断是否属于音准问题，以及当前标注是否覆盖该类型。",
         pyin_key="031_candidate0", expected=65),
    dict(sample="007", start=23.12, end=23.24, clip_start=22.6, clip_end=23.75,
         title="空标注的低置信候选：泛音/过吹歧义",
         finding="序列比对的预期音高为 MIDI 50，但基频估计约 MIDI 68.8，接近第 3 泛音。单簧管的泛音结构可能使估计器误判，也可能确有过吹，因此只能列作复听候选，不能自动补标错音。",
         pyin_key="007_candidate1", expected=50),
    dict(sample="035", start=8.0795, end=9.5675, clip_start=6.8, clip_end=10.5,
         title="重复标注缺少 canonical 评分所需信息",
         finding="两条 repetition 均没有 extra_copies 和 repeats_label_range。缺少字段不等于人耳判断错误，但目前不能用它们完成要求重复来源/次数的 canonical 评估。"),
]

(HERE / "clips").mkdir(exist_ok=True)
(HERE / "figures").mkdir(exist_ok=True)
cards = []
for k, case in enumerate(cases):
    sid = case["sample"]
    source = HERE.parent / "bundles" / sid / "performance_audio.wav"
    info = sf.info(source)
    lo, hi = case["clip_start"], min(case["clip_end"], info.duration)
    audio, sr = sf.read(source, start=int(lo * info.samplerate), stop=int(hi * info.samplerate))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    basename = f"{k + 1:02d}_{sid}"
    clip_path = HERE / "clips" / (basename + ".wav")
    sf.write(clip_path, audio, sr, subtype="PCM_16")
    times = lo + np.arange(len(audio)) / sr
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True, layout="constrained")
    skip = max(1, len(audio) // 6000)
    axes[0].plot(times[::skip], audio[::skip], lw=.65, color="#335e7a")
    axes[0].set_ylabel("Waveform")
    track = np.load(HERE / "pitch" / (sid + ".npz"))
    select = (track["time"] >= lo) & (track["time"] <= hi) & track["voiced"]
    axes[1].scatter(track["time"][select], track["midi"][select], s=7, alpha=.45, label="YIN")
    if case.get("pyin_key"):
        p = np.load(HERE / "pyin" / (case["pyin_key"] + ".npz"))
        select = p["voiced"] & (p["probability"] >= .5) & np.isfinite(p["midi"])
        axes[1].scatter(p["time"][select], p["midi"][select], s=9, color="#ed963c", label="pYIN p>=0.5")
    if "expected" in case:
        axes[1].hlines(case["expected"], case["start"], case["end"], color="#b44343", lw=2, label="Selected score pitch")
    for ax in axes:
        ax.axvspan(case["start"], case["end"], color="#e9c94a", alpha=.2)
        ax.set_xlim(lo, hi);ax.grid(alpha=.15)
    axes[1].set_ylabel("Sounding MIDI pitch");axes[1].set_xlabel("Original recording time (s)")
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Sample {sid}: {case['start']:.4f}-{case['end']:.4f} s")
    plot_path = HERE / "figures" / (basename + ".png")
    fig.savefig(plot_path, dpi=140);plt.close(fig)
    image_url = "data:image/png;base64," + base64.b64encode(plot_path.read_bytes()).decode()
    audio_url = "data:audio/wav;base64," + base64.b64encode(clip_path.read_bytes()).decode()
    cards.append(f'''<article><h2>{sid} · {html.escape(case['title'])}</h2>
<p>{html.escape(case['finding'])}</p><p class="meta">录音原时间 {lo:.3f}–{hi:.3f} 秒；黄色为待复核区间。音频未增益、未改调。</p>
<audio controls preload="none" src="{audio_url}"></audio><img src="{image_url}" alt="Waveform and pitch evidence"></article>''')

counts = Counter(issue for v in structure.values() for l in v["labels"] for issue in l["issues"])
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>034 数据标注复核</title><style>body{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#172b3a}main{max-width:1000px;margin:auto;padding:28px}article,.intro{background:white;padding:24px;margin-bottom:20px;border-radius:12px;border:1px solid #dbe2e7}h1{font-size:28px}h2{font-size:20px}audio{width:100%}img{width:100%;height:auto}.meta{font-size:13px;color:#536a7b}.notice{border-left:4px solid #cf9721;padding-left:14px}</style><main>
<h1>修正后 40 条录音：标注复核</h1><section class="intro"><p>检查范围：40 份标签、63 条标注的文件与谱面核对；40 条音频的自动基频初筛；36 个重点窗口的 pYIN 复查。</p>
<p class="notice">这是声学辅助核查，未进行逐条人工听审。YIN 和 pYIN 同属一个算法家族，不是独立人类标注者。无稳定基频不等于没有声音；音高估计异常不自动等于漏标。原始标签和正式测试结果均未改动。</p>
<p>确定的问题：002 的四条完全重复记录；005 的同类嵌套覆盖。另有 23 条存储音高列表与当前谱面范围不符、5 条 UI 核心选择与核心范围不符、2 条重复标注缺少 canonical 所需字段。三类问题可重叠，不能相加当作“标错条数”。</p>
<p>core_note_ids 使用包含休止符的 UI 事件 ID，score_part 使用发声音符索引；本次经过真实映射核对，没有把编号天然不同当成错误。</p></section>'''
page += "\n".join(cards) + "</main></html>"
(HERE / "review.html").write_text(page)

with (HERE / "label_review.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["sample", "label_index", "label_id", "type", "start", "end", "structural_findings", "exact_duplicate_group", "pyin_median_sounding_midi", "pyin_reliable_frames", "perceptual_status"])
    for sid, item in structure.items():
        for label in item["labels"]:
            p = checks.get(f"{sid}_label{label['index']}", {})
            duplicate = next((g for g in item["exact_duplicate_groups"] if label["index"] in g), [])
            writer.writerow([sid, label["index"], label["id"], label["type"], label["start"], label["end"],
                ";".join(label["issues"]), str(duplicate) if duplicate else "", p.get("midi_median", ""),
                p.get("reliable_frames", ""), "not human-adjudicated"])

integrity = {}
for sid, item in structure.items():
    digest = hashlib.sha256((HERE.parent / "bundles" / sid / "labels.json").read_bytes()).hexdigest()
    assert digest == item["label_file_sha256"]
    integrity[sid] = digest
(HERE / "integrity.json").write_text(json.dumps(dict(all_40_original_labels_unchanged=True,
    labels_sha256=integrity,scope=dict(label_documents=40,labels=63,pitch_screened_clips=40,pyin_windows=len(checks)),
    findings=dict(counts),human_listening_performed=False), indent=2) + "\n")
print(HERE / "review.html")
