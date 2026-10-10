"""V10 UI contract, retaining the established canonical note mapping."""
from .datacreate_v9 import gui_documents as v9_documents


def gui_documents(*args, **kwargs):
    gui, labels = v9_documents(*args, **kwargs)
    gui['summary'].update(backend='ALIGN v10 (experimental)',
        candidate_generation='dual-mel CTC + guarded same-pitch-v3',
        review_notice='V10 restores acoustically supported repeated notes while preserving other mappings.')
    gui['label_generation'].update(method='align_stack_v10', annotator_id='align_stack_v10_review')
    labels['annotator_id'] = 'align_stack_v10_review'
    labels['agent_labeling']['method'] = 'align_stack_v10'
    for label in gui['labels']:
        label['comment'] = 'ALIGN v10: guarded same-pitch boundary repair; reference-note identity is primary.'
    return gui, labels
