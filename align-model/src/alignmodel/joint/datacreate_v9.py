"""Adapt v9 results to DataCreate's existing annotation/alignment UI contract."""
from __future__ import annotations
import math
from datacreate.melody import parse_sounding_notes
from datacreate.models import LabelsDocument
from datacreate.note_alignment import _midi_pitch_name


def gui_documents(sample, index, alignment, feedback, *, duration, provenance):
    sounding=parse_sounding_notes(sample/'verified_score.musicxml')
    # Both parsers collapse explicit ties. Refuse to silently shift note identity.
    if len(sounding)!=len(index.events) or any(
        n.pitch!=e.pitch or abs(n.ql_start-e.ql_start)>1e-5 or abs(n.ql_end-e.ql_end)>1e-5
        for n,e in zip(sounding,index.events)):
        raise ValueError(f'{sample.name}: canonical/UI score-note index mismatch')
    info=feedback['diagnostics'];repair=info['same_pitch_repair']
    transcribed=[];mapping=[None]*len(alignment.notes);events=[]
    kept=set(alignment.kept_note_indices)
    for i,n in enumerate(alignment.notes):
        transcribed.append({'pitch':int(n.pitch),'start':float(n.start),'end':float(n.end),
            'confidence':float(n.confidence),'optional':bool(n.optional),
            'source_candidate_indices':repair['source_groups'][i],
            'ignored':i not in kept,'ignored_reason':'aligner_dropped_candidate' if i not in kept else None})
    # Preserve raw alignment for inspection even when error feedback abstains.
    for e in alignment.events:
        row=alignment.kept_note_indices[e.rendered_index]
        transcribed[row].update(alignment_pitch=e.pitch,relationship=e.relationship,
                                score_span=list(e.score_span) if e.score_span else None)
        if not e.score_span:continue
        mapping[row]=e.score_span[0]
        for si in range(*e.score_span):
            note=sounding[si]
            events.append({'id':f'aligned_{len(events):05d}','note_id':note.note_id,
                'sounding_index':si,'score_index':si,'transcription_index':row,
                'is_rest':False,'pitch':_midi_pitch_name(note.pitch),'midi':note.pitch,'measure':note.measure,
                'duration_ql':note.ql_end-note.ql_start,'ref_start':note.start,'ref_end':note.end,
                'perf_start':e.start,'perf_end':e.end,'alignment_kind':e.relationship,
                'is_repetition':bool(e.copy_pass),'alignment_status':feedback['status']})
    labels=[];unavailable=[];score_only=[]
    for raw in feedback['labels']:
        ids=raw['score_event_indices']
        if any(i<0 or i>=len(sounding) for i in ids):raise ValueError('invalid label identity')
        a,b=raw.get('start_time'),raw.get('end_time')
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            selected=[sounding[i] for i in ids]
            score_only.append({**raw,'source':'agent','timing_status':'unavailable',
                'start_time':float(a) if a is not None and math.isfinite(a) else None,
                'end_time':float(b) if b is not None and math.isfinite(b) else None,
                'note_id':selected[0].note_id,'note_ids':[n.note_id for n in selected],
                'core_note_ids':[n.note_id for n in selected],'measure_number':selected[0].measure})
            unavailable.append(raw['id']);continue
        # Playback regions must be inside the recording. Identity is unchanged.
        start=max(0.,min(float(a),max(0.,duration-.001)))
        end=max(start+.001,min(float(b),duration))
        selected=[sounding[i] for i in ids]
        label={**raw,'source':'agent','start_time':start,'end_time':end,'severity':2,
            'note_id':selected[0].note_id,'note_ids':[n.note_id for n in selected],
            'core_note_ids':[n.note_id for n in selected],'pitches':[n.pitch for n in selected],
            'measure_number':selected[0].measure,
            'comment':'ALIGN v9 experimental: relaxed same-pitch repair; reference-note identity is primary.',
            'score_part':{**raw['score_part'],'start_measure':selected[0].measure,'end_measure':selected[-1].measure}}
        if raw.get('repeats_label_range'):
            r=raw['repeats_label_range']
            if r.get('start_time') is None or r.get('end_time') is None:label.pop('repeats_label_range')
        labels.append(label)
    summary={'engine':'align-joint','backend':'ALIGN v9 (experimental)',
        'pipeline_revision':info.get('pipeline_revision'),
        'passage_location':info.get('passage_location'),
        'candidate_generation':'dual-mel CTC + same-pitch-v2',
        'event_count':len(events),'transcribed_note_count':len(transcribed),
        'mapped_note_count':sum(i is not None for i in mapping),
        'kept_note_count':len(kept),'ignored_note_count':len(transcribed)-len(kept),
        'score_event_count':len(sounding),'candidate_count':len(labels),
        'status':feedback['status'],'match_fraction':alignment.match_fraction,
        'same_pitch_merged':repair['merged_boundaries'],'same_pitch_pairs':repair['pairs_checked'],
        'note_end_policy':'Decoder next-onset estimates; not measured acoustic offsets.',
        'review_notice':('Alignment uncertain: raw hypothesis shown for review; error labels withheld.'
                         if feedback['status']!='ok' else 'Experimental v9: check merged repeated notes.')}
    payload={'format_version':2,'engine':'align-joint','sample_id':sample.name,
        'label_generation':{'schema_version':'datacreate-model-feedback-v1',
                            'method':'align_stack_v9','annotator_id':'align_stack_v9_review'},
        'events':events,'labels':labels,'score_only_labels':score_only,'transcribed_notes':transcribed,'note_mapping':mapping,
        'repetitions':[l for l in labels if l['type']=='repetition'],
        'summary':summary,'provenance':provenance,'diagnostics':info,
        'unassessed_score_event_indices':feedback['unassessed_score_event_indices']}
    document={'schema_version':'1.2','audio_reference':'performance_audio.wav',
        'annotator_id':'align_stack_v9_review','self_reported':[],'labels':labels,
        'agent_labeling':{**provenance,'method':'align_stack_v9','status':feedback['status'],
                          'labels_without_playback_time':unavailable,'score_only_labels':score_only,'training_performed':False}}
    LabelsDocument.model_validate(document)
    return payload,document
