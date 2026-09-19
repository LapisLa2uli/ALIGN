"""Build a standalone, read-only comparison from frozen sample 004 outputs."""
import base64
import collections
import io
import json
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pretty_midi
import soundfile as sf
from PIL import Image

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parents[1]
BUNDLE = RUN / 'bundles/004'


def audio(name):
    path = BUNDLE / name
    return {'url': 'data:audio/wav;base64,' + base64.b64encode(path.read_bytes()).decode(),
            'duration': sf.info(path).duration}


def midi(path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted([{'type': ins.name, 'start_time': n.start, 'end_time': n.end,
                    'pitch': n.pitch, 'note': pretty_midi.note_number_to_name(n.pitch)}
                   for ins in pm.instruments for n in ins.notes],
                  key=lambda n: (n['start_time'], n['type'], n['pitch']))


def melgrams():
    # Recompute from the embedded WAVs: cached arrays can belong to an older export.
    sr, hop, n_fft = 22050, 512, 2048
    powers = {}
    for name in ['performance', 'reference']:
        samples, source_sr = sf.read(BUNDLE / f'{name}_audio.wav', always_2d=True)
        samples = samples.mean(axis=1)
        if source_sr != sr:
            samples = librosa.resample(samples, orig_sr=source_sr, target_sr=sr)
        powers[name] = librosa.feature.melspectrogram(
            y=samples, sr=sr, n_fft=n_fft, hop_length=hop,
            n_mels=128, fmin=30, fmax=sr / 2, power=2.0)
    peak = max(float(p.max()) for p in powers.values())
    result = {}
    for name, power in powers.items():
        db = np.clip(10 * np.log10(np.maximum(power, 1e-20) / peak), -80, 0)
        pixels = matplotlib.colormaps['magma']((db[::-1] + 80) / 80, bytes=True)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format='PNG')
        (OUT / f'{name}_mel.png').write_bytes(buffer.getvalue())
        result[name] = {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                        'start': -hop / (2 * sr), 'duration': power.shape[1] * hop / sr}
    lo, hi = librosa.hz_to_mel(np.array([30, sr / 2]))
    ticks = [{'hz': f, 'fraction': float((librosa.hz_to_mel(f) - lo) / (hi - lo))}
             for f in [100, 500, 1000, 2000, 4000, 8000]]
    return {'images': result, 'ticks': ticks}


data = {'sample': '004', 'performance': audio('performance_audio.wav'),
        'reference': audio('reference_audio.wav'),
        'reference_notes': midi(BUNDLE / 'reference_audio.mid'),
        'gold': json.loads((BUNDLE / 'labels.json').read_text())['labels'], 'models': {}}
data['mel'] = melgrams()
assert data['gold'] == []
for model in ['polytune', 'laddersym']:
    labels = json.loads((RUN / f'{model}_frozen/004.json').read_text())['labels']
    native_path = ROOT / f'baselines/runs/{model}/eval_real034_20260918_full40/real034_20260918_full40/004/mix.mid'
    native = midi(native_path)
    data['models'][model] = {'labels': labels, 'native': native,
                             'counts': dict(collections.Counter(x['type'] for x in labels))}
    (OUT / f'{model}.lab').write_text(''.join(
        f"{x['start_time']:.9f}\t{x['end_time']:.9f}\t{x['type']}\n" for x in labels))
(OUT / 'reference_gold.lab').write_text('')
(OUT / 'comparison.json').write_text(json.dumps({k:v for k,v in data.items()
                                                if k not in ('performance','reference','mel')}, indent=2))

template = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>004 · 人工标签与 Baselines 对照</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,"Noto Sans SC",sans-serif;color:#182b40;background:#f2f5f9}*{box-sizing:border-box}body{max-width:1220px;margin:auto;padding:28px}h1{font-size:28px;margin-bottom:8px}h2{font-size:19px;margin-top:0}p{line-height:1.75}.sub,.muted{color:#526478;font-size:14px}.card{background:white;border:1px solid #dce3ec;border-radius:14px;padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.stat{border-left:4px solid #627a97;padding:5px 16px}.stat strong{font-size:30px;display:block}.gold{border-color:#219675}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:15px 0}button,select,input{font:inherit;border:1px solid #c7d3e0;border-radius:7px;padding:7px;background:white}button{cursor:pointer}button:hover{background:#eaf1f8}audio{width:100%;max-width:520px}input[type=range]{flex:1;min-width:150px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}svg{display:block;width:100%;background:#fafcff;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e4eaf1;padding:9px}th{background:#f5f8fc;position:sticky;top:0}.scroll{max-height:420px;overflow:auto}.note{background:#eef5fb;border-radius:8px;padding:12px 16px;font-size:14px}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:24px}code{font-size:13px}a{color:#2263a8}#hover{min-height:25px;font-size:14px;color:#43586f}@media(max-width:700px){body{padding:12px}.grid,.twocol{grid-template-columns:1fr}.card{padding:14px}}
</style>
<h1>004：人工标签与两个 Baseline 的对照</h1>
<p class="sub">034.zip 测试集 · 已采用 036–040 修正版 · 使用正式评测冻结的原始预测，没有使用音量调整对照实验的输出。</p>
<div class="card"><div class="grid" id="stats"></div>
<p>这条录音的人工真值为 <code>labels: []</code>，按已确认的标注约定表示<strong>没有演奏错误</strong>。因此，下方所有预测错误事件，在当前真值下都计为误报。这个例子用于展示误报，并不代表所有样本。</p>
<p class="note">这里的 Reference gold 是人工错误标签；Reference audio / MIDI 是模型接收的参考演奏。两者含义不同。参考音频与真实录音的速度、起始时间不同，下方分别使用各自时间轴，不把同一秒强行当作同一音符。</p></div>
<div class="card"><h2>1. 演奏录音上的错误标签</h2><audio id="perf" controls preload="metadata"></audio>
<div class="controls"><button id="focus">看 4–8 秒</button><button id="whole">看整条录音</button><label>窗口 <select id="window"><option value="4">4 秒</option><option value="8" selected>8 秒</option><option value="16">16 秒</option><option value="all">全部</option></select></label><span id="rangeText"></span></div>
<div class="controls"><label for="start">窗口起点</label><input id="start" type="range" min="0" step="0.01" value="0"></div>
<h2>演奏 Mel spectrogram</h2>
<p class="muted">频谱与下方错误标签共用时间窗口和横轴；拖动窗口或点击频谱试听。颜色由暗到亮表示能量由低到高。</p>
<svg id="perfMel" viewBox="0 0 1100 255" role="img" aria-label="演奏录音梅尔频谱"></svg>
<p class="muted">两张频谱采用相同色标：−80 至 0 dB，以两段音频共同的最大梅尔功率为 0 dB；没有分别归一化音量。深色 → 紫红 → 橙黄 → 浅黄。仅用于查看，不是模型内部特征。</p>
<div class="legend" id="legend"></div><p class="muted">点击时间轴定位录音；悬停查看精确时间。同一轨内上下错开表示事件重叠，不表示音高。极短事件显示至少 2 像素。</p>
<svg id="timeline" viewBox="0 0 1100 280" role="img" aria-label="人工与两个模型的错误标签时间轴"></svg><div id="hover"></div>
<div class="controls"><button data-lab="gold">下载人工 reference.lab（空文件）</button><button data-lab="polytune">下载 Polytune.lab</button><button data-lab="laddersym">下载 LadderSym.lab</button></div>
<p class="muted">LAB 格式：起始秒 TAB 结束秒 TAB 错误类型。标签由模型原生 MIDI 经正式评测的固定转换规则得到；wrong_note、repetition 等并非全部由模型直接输出。重复行保留，未为展示合并或删除。</p>
<div class="controls"><label>标签明细 <select id="which"><option value="polytune">Polytune</option><option value="laddersym">LadderSym</option><option value="gold">人工 Reference</option></select></label><label><input id="visibleOnly" type="checkbox" checked>仅当前窗口中有重叠的事件</label><span id="rowCount" class="muted"></span></div>
<div class="scroll"><table><thead><tr><th>#</th><th>开始 / s</th><th>结束 / s</th><th>标签</th><th>人工真值对照</th><th>试听</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="card"><h2>2. 参考音频与参考 MIDI</h2><audio id="ref" controls preload="metadata"></audio>
<h2 style="margin-top:20px">参考 Mel spectrogram</h2>
<svg id="refMel" viewBox="0 0 1100 255" role="img" aria-label="参考音频梅尔频谱"></svg>
<p class="muted">参考频谱采用参考音频自身的完整时间轴，与下方 MIDI 横轴对应；点击频谱定位参考音频。频谱从本页 WAV 重新计算：22,050 Hz、FFT 2,048、hop 512、128 个 Mel 频带、30–11,025 Hz。</p>
<p class="muted">112 个参考 MIDI 音符；音高直接读取模型输入 MIDI（实际发声音高），不使用移调乐器谱面的书写音高。点击下图可定位参考音频。</p>
<svg id="referencePlot" viewBox="0 0 1100 230" role="img" aria-label="参考 MIDI 音高与时间"></svg>
<details><summary>展开参考音符列表（参考音频自身时间轴）</summary><div class="scroll"><table><thead><tr><th>#</th><th>开始 / s</th><th>结束 / s</th><th>音名</th><th>MIDI</th></tr></thead><tbody id="refRows"></tbody></table></div></details></div>
<div class="card"><h2>3. 模型原生输出：音高及类别</h2><p class="muted">这里直接读取正式推理输出的 mix.mid，保留 extra / missing / correct。它与上方转换后的错误事件数不同：转换会配对错音、合并重复片段，并移除 correct。原生时间为模型输出时间，不是经过人工确认的音符对齐。</p><div class="twocol" id="native"></div></div>
<div class="card"><h2>如何理解这个例子</h2><p>人工标签认为整条录音无错误，而两个模型都连续输出 extra（多音）等错误标签，因此这个例子的主要问题是大量误报。仅凭这些标签无法证明模型为何出错，也不能认定每个预测音高都转录错误。</p><p class="muted">来源：bundles/004/labels.json、reference_audio.mid、performance_audio.wav、reference_audio.wav；polytune_frozen/004.json、laddersym_frozen/004.json；两个模型的 eval_real034_20260918_full40/real034_20260918_full40/004/mix.mid。所有音频、标签与音符数据均已嵌入此 HTML，可以离线打开。本页面没有修改任何原始标注或正式评测结果。</p></div>
<script>
const D=__DATA__;
const colors={extra_note:'#e09a30',wrong_note:'#d65366',missed_note:'#8a66bd',repetition:'#328bb4',rhythm_error:'#888'};
const names={extra_note:'多音',wrong_note:'错音',missed_note:'漏音',repetition:'重复',rhythm_error:'节奏错误'};
const $=id=>document.getElementById(id), fmt=n=>n.toFixed(3);
const perf=$('perf'),ref=$('ref');perf.src=D.performance.url;ref.src=D.reference.url;
perf.onplay=()=>ref.pause();ref.onplay=()=>perf.pause();
$('stats').innerHTML='<div class="stat gold">人工 Reference<strong>0</strong>错误事件</div>'+Object.entries(D.models).map(([m,d])=>`<div class="stat">${m==='polytune'?'Polytune':'LadderSym'}<strong>${d.labels.length}</strong>${Object.entries(d.counts).map(([k,n])=>names[k]+' '+n).join(' · ')}</div>`).join('');
$('legend').innerHTML=Object.entries(colors).filter(([k])=>k!=='rhythm_error').map(([k,c])=>`<span><i class="dot" style="background:${c}"></i>${names[k]} / ${k}</span>`).join('');
function bounds(){let w=$('window').value==='all'?D.performance.duration:Number($('window').value);w=Math.min(w,D.performance.duration);$('start').max=Math.max(0,D.performance.duration-w);let a=Math.min(Number($('start').value),Number($('start').max));return[a,a+w]}
function labels(m){return m==='gold'?D.gold:D.models[m].labels}
function seek(audio,t){audio.currentTime=Math.max(0,t);audio.play().catch(()=>{})}
function melPlot(id,name,a,b,left){const right=1080,top=26,height=195,x=t=>left+(t-a)/(b-a)*(right-left),im=D.mel.images[name],audio=name==='performance'?perf:ref;let s=`<defs><clipPath id="${id}Clip"><rect x="${left}" y="${top}" width="${right-left}" height="${height}"/></clipPath></defs><g clip-path="url(#${id}Clip)"><image href="${im.url}" x="${x(im.start)}" y="${top}" width="${im.duration/(b-a)*(right-left)}" height="${height}" preserveAspectRatio="none"/></g>`;
for(let i=0;i<=8;i++){const t=a+(b-a)*i/8;s+=`<text x="${x(t)}" y="16" text-anchor="middle" font-size="12" fill="#63758a">${t.toFixed(1)}s</text>`}
for(const tick of D.mel.ticks){const y=top+height*(1-tick.fraction);s+=`<text x="${left-8}" y="${y+4}" text-anchor="end" font-size="12" fill="#63758a">${tick.hz>=1000?tick.hz/1000+'k':tick.hz} Hz</text>`}
s+=`<line id="${id}Cursor" x1="${x(audio.currentTime)}" x2="${x(audio.currentTime)}" y1="${top}" y2="${top+height}" stroke="white" stroke-width="2"/><text x="${left}" y="245" font-size="12" fill="#63758a">${name==='performance'?'演奏时间':'参考时间'} / s · Mel 频率轴</text>`;$(id).innerHTML=s;
$(id).onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*1100;if(px>=left&&px<=right)seek(audio,a+(px-left)/(right-left)*(b-a))};}
function melCursor(id,t,a,b,left){const c=$(id+'Cursor'),x=left+(t-a)/(b-a)*(1080-left);if(c){c.setAttribute('x1',x);c.setAttribute('x2',x);c.style.display=x<left||x>1080?'none':''}}
function render(){const[a,b]=bounds(),left=150,right=1080,x=t=>left+(t-a)/(b-a)*(right-left);let s='',y=36;
melPlot('perfMel','performance',a,b,150);
for(let i=0;i<=8;i++){let t=a+(b-a)*i/8;s+=`<text x="${x(t)}" y="19" text-anchor="middle" font-size="12" fill="#63758a">${t.toFixed(1)}s</text>`}
for(const[m,title]of [['gold','人工 Reference'],['polytune','Polytune'],['laddersym','LadderSym']]){const ends=[];let bars='';for(const[eid,e]of labels(m).entries()){if(e.end_time<=a||e.start_time>=b)continue;let lane=ends.findIndex(end=>end<=e.start_time);if(lane<0)lane=ends.length;ends[lane]=e.end_time;let xx=x(Math.max(a,e.start_time)),ww=Math.max(2,x(Math.min(b,e.end_time))-xx);bars+=`<rect class="event" data-info="${title} #${eid+1}: ${e.type} · ${e.start_time.toFixed(6)}–${e.end_time.toFixed(6)} s" x="${xx}" y="${y+lane*16}" width="${ww}" height="12" rx="2" fill="${colors[e.type]||'#888'}"><title>${e.type} ${fmt(e.start_time)}–${fmt(e.end_time)}s</title></rect>`}
let h=Math.max(44,ends.length*16+14);s+=`<rect x="${left}" y="${y-4}" width="${right-left}" height="${h}" fill="${m==='gold'?'#edf8f3':'#f0f4f9'}"/><text x="8" y="${y+15}" font-size="14">${title}</text>`+bars;if(m==='gold')s+=`<text x="${left+12}" y="${y+18}" font-size="13" fill="#27765b">无错误标签</text>`;y+=h+20}
s+=`<line id="cursor" x1="${x(perf.currentTime)}" x2="${x(perf.currentTime)}" y1="28" y2="${y}" stroke="#172f4e" stroke-width="1.5"/>`;
$('timeline').setAttribute('viewBox',`0 0 1100 ${y+5}`);$('timeline').innerHTML=s;$('rangeText').textContent=`${fmt(a)}–${fmt(b)} 秒`;
document.querySelectorAll('.event').forEach(el=>{el.onmouseenter=()=>$('hover').textContent=el.dataset.info;el.onmouseleave=()=>$('hover').textContent=''});renderRows();}
function renderRows(){const[a,b]=bounds(),m=$('which').value,filtered=labels(m).map((e,i)=>({...e,i})).filter(e=>!$('visibleOnly').checked||(e.end_time>a&&e.start_time<b));$('rowCount').textContent=`显示 ${filtered.length} / ${labels(m).length} 条`;$('rows').innerHTML=filtered.length?filtered.map(e=>`<tr><td>${e.i+1}</td><td>${fmt(e.start_time)}</td><td>${fmt(e.end_time)}</td><td>${names[e.type]} <code>${e.type}</code></td><td>无错误 → 误报</td><td><button data-seek="${e.start_time}">播放</button></td></tr>`).join(''):'<tr><td colspan="6">没有错误标签。</td></tr>';document.querySelectorAll('[data-seek]').forEach(el=>el.onclick=()=>seek(perf,Number(el.dataset.seek)));}
$('start').oninput=render;$('window').onchange=render;$('which').onchange=renderRows;$('visibleOnly').onchange=renderRows;
$('focus').onclick=()=>{$('window').value='4';$('start').value=4;render();seek(perf,4)};$('whole').onclick=()=>{$('window').value='all';$('start').value=0;render()};
$('timeline').onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*1100,[a,b]=bounds();if(px>=150&&px<=1080)seek(perf,a+(px-150)/930*(b-a))};
perf.ontimeupdate=()=>{const[a,b]=bounds(),c=$('cursor'),x=150+(perf.currentTime-a)/(b-a)*930;if(c){c.setAttribute('x1',x);c.setAttribute('x2',x);c.style.display=x<150||x>1080?'none':''}melCursor('perfMel',perf.currentTime,a,b,150)};
document.querySelectorAll('[data-lab]').forEach(el=>el.onclick=()=>{const m=el.dataset.lab,txt=labels(m).map(e=>`${e.start_time.toFixed(9)}\t${e.end_time.toFixed(9)}\t${e.type}\n`).join(''),url=URL.createObjectURL(new Blob([txt],{type:'text/plain;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`004_${m==='gold'?'reference_gold':m}.lab`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
const rn=D.reference_notes,min=Math.min(...rn.map(n=>n.pitch))-2,max=Math.max(...rn.map(n=>n.pitch))+2,rx=t=>60+t/D.reference.duration*1020,ry=p=>195-(p-min)/(max-min)*160;let rs='';
for(let p=Math.ceil(min/12)*12;p<=max;p+=12)rs+=`<line x1="60" x2="1080" y1="${ry(p)}" y2="${ry(p)}" stroke="#dce3ec"/><text x="5" y="${ry(p)+4}" font-size="12">MIDI ${p}</text>`;
for(let t=0;t<D.reference.duration;t+=5)rs+=`<text x="${rx(t)}" y="222" font-size="12">${t}s</text>`;
rs+=rn.map(n=>`<rect x="${rx(n.start_time)}" y="${ry(n.pitch)}" width="${Math.max(2,rx(n.end_time)-rx(n.start_time))}" height="4" fill="#32877a"><title>${n.note} (${n.pitch}) · ${fmt(n.start_time)}–${fmt(n.end_time)}s</title></rect>`).join('');rs+='<line id="refCursor" x1="60" x2="60" y1="25" y2="200" stroke="#172f4e"/>';$('referencePlot').innerHTML=rs;
$('referencePlot').onclick=e=>{const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*1100;if(x>=60&&x<=1080)seek(ref,(x-60)/1020*D.reference.duration)};ref.ontimeupdate=()=>{let x=rx(ref.currentTime);$('refCursor').setAttribute('x1',x);$('refCursor').setAttribute('x2',x);melCursor('refMel',ref.currentTime,0,D.reference.duration,60)};
melPlot('refMel','reference',0,D.reference.duration,60);
$('refRows').innerHTML=rn.map((n,i)=>`<tr><td>${i+1}</td><td>${fmt(n.start_time)}</td><td>${fmt(n.end_time)}</td><td>${n.note}</td><td>${n.pitch}</td></tr>`).join('');
$('native').innerHTML=Object.entries(D.models).map(([m,d])=>{const counts={};d.native.forEach(n=>counts[n.type]=(counts[n.type]||0)+1);return `<div><h2>${m==='polytune'?'Polytune':'LadderSym'}</h2><p class="muted">${Object.entries(counts).map(([k,n])=>`${k}: ${n}`).join(' · ')}</p><div class="scroll"><table><thead><tr><th>开始 / s</th><th>结束 / s</th><th>音高</th><th>原生类别</th></tr></thead><tbody>${d.native.map(n=>`<tr><td>${fmt(n.start_time)}</td><td>${fmt(n.end_time)}</td><td>${n.note} (${n.pitch})</td><td>${n.type}</td></tr>`).join('')}</tbody></table></div></div>`}).join('');
render();
</script></html>'''
(OUT / 'comparison.html').write_text(template.replace('__DATA__', json.dumps(data)), encoding='utf-8')
print(json.dumps({'html': str(OUT / 'comparison.html'), 'bytes': (OUT / 'comparison.html').stat().st_size,
                  'events': {m:len(d['labels']) for m,d in data['models'].items()},
                  'reference_notes': len(data['reference_notes'])}, indent=2))
