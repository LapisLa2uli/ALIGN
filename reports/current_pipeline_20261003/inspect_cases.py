import json,sys
from pathlib import Path
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here))
import run_audit as a
from plot_datacreate_predictions import plot
for take in ('004','034','020'):
    alignment=a.read(here/'alignments/datacreate'/f'{take}.json')
    events=sorted(alignment['events'],key=lambda e:e['note_index'])
    if take=='020':
        selected=min(range(len(events)),key=lambda k:abs(events[k]['start']-12.558))
    else:
        selected=min((k for k,e in enumerate(events) if e['relationship']=='extra'),key=lambda k:events[k]['end']-events[k]['start'])
    plot(a.ROOT/'DataCreate/samples'/take,alignment,selected,here/'figures'/f'{take}.png',f'Take {take}: model output near {events[selected]["start"]:.3f} seconds')
d=a.read(here/'datacreate.json')
for row in d['per_sample']:
    if any(g['type']=='repetition' for g in row['gold']):
        print('REPEAT',row['sample'],json.dumps([g for g in row['gold'] if g['type']=='repetition']),json.dumps([p for p in row['predicted'] if p['type']=='repetition']))
s=a.read(here/'synthetic.json')
print('ERRORS',s['errors'])
print('GATES',s['datacreate_gate_totals'])
print('EXTRAS',a.read(here/'diagnostics.json')['extra_event_stats'])
print('SYNTH_CASES',json.dumps([r for r in s['details'] if r.get('missed_removed_by_gate')][:3]))
