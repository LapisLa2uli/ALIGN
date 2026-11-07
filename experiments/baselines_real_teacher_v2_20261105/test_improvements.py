import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from common import ROOT,RUN,PREVIOUS,read,split
sys.path.insert(0,str(ROOT/'baselines/common'))
from canonical_adapter_v2 import locate_extra_neighbors,adapt,Note
from canonical_adapter import adapt as old_adapt
from reference_alignment import track_cost,context_start,crop

def event(kind,index=None,copy_pass=0):return {'kind':kind,'score_index':index,'copy_pass':copy_pass}

class Improvements(unittest.TestCase):
    def test_extra_anchors_boundaries_and_no_evidence(self):
        e=[event('correct',2),event('extra'),event('correct',3)]
        self.assertEqual(locate_extra_neighbors(e,8)[1]['score_event_indices'],[2,3])
        self.assertEqual(locate_extra_neighbors([event('extra'),event('correct',0)],8)[0]['score_event_indices'],[0,1])
        self.assertEqual(locate_extra_neighbors([event('correct',7),event('extra')],8)[1]['score_event_indices'],[7])
        self.assertIsNone(locate_extra_neighbors([event('extra')],8)[0]['score_index'])

    def test_extra_does_not_bridge_backwards_or_copy_boundary(self):
        for left,right in [(event('correct',6),event('correct',2)),(event('copy',2,1),event('correct',3))]:
            self.assertIsNone(locate_extra_neighbors([left,event('extra'),right],10)[1]['score_index'])

    def test_other_native_classes_unchanged(self):
        notes=[Note(0,.2,60,'correct'),Note(.3,.5,64,'extra'),Note(.3,.5,62,'missing'),Note(.6,.8,65,'missing')]
        a=old_adapt(notes,[60,62,65]);b=adapt(notes,[60,62,65]);self.assertEqual(a['labels'],b['labels'])

    def test_path_follows_slow_fast_and_repeat(self):
        expected=list(range(15))+list(range(5,15))+[15,15,16,18,20,22]
        cost=np.full((25,len(expected)),20,dtype='float32')
        cost[expected,np.arange(len(expected))]=0
        np.testing.assert_array_equal(track_cost(cost),expected)

    def test_context_chooses_passage_not_average_at_jump(self):
        mapping={'performance_times':np.arange(30)*.032,'reference_times':np.r_[10+np.arange(20)*.032,2+np.arange(10)*.032]}
        self.assertGreater(context_start(mapping,0,.93),7)
        np.testing.assert_array_equal(crop(np.arange(5),-2,5,sr=1),[0,0,0,1,2])

    def test_cached_train_inference_frontend_parity(self):
        for m in ['polytune','laddersym']:
            manifest=read(RUN/'cache'/m/'manifest.json')
            self.assertTrue(all(v['performance_parity_max_abs']<1e-5 for v in manifest.values()))
            for s in split()['train']:
                self.assertEqual(manifest['train/'+s]['bounds'],read(PREVIOUS/'cache'/m/f'{s}.json')['bounds'])

    def test_prompt_matches_native_tokenizer_and_silence(self):
        from features import Features
        import note_seq
        f=Features('laddersym');ns=note_seq.NoteSequence()
        for start,pitch in [(.2,60),(2.8,65)]:
            n=ns.notes.add();n.start_time=start;n.end_time=start+.2;n.pitch=pitch;n.velocity=100
        ns.total_time=3.
        prompt,mask=f.prompt(ns,0)
        row=f.ds._tokenize([ns,note_seq.NoteSequence(),note_seq.NoteSequence(),ns],np.zeros(32768,dtype='float32'),np.zeros(65536,dtype='float32'))
        row=f.ds._extract_target_sequence_with_indices(row,f.ds.tie_token)
        row=f.ds.run_length_encode_shifts(row,feature_key='prompt_events')
        expected=f.ds._pad_length({'prompt_events':row['prompt_events']})
        import torch
        self.assertTrue(torch.equal(prompt[0],expected['prompts']))
        silent,silentmask=f.prompt(note_seq.NoteSequence(),0)
        self.assertEqual(silent.shape,(1,1024));self.assertGreater(int(silentmask.sum()),0)
        # Notes in the second half affect both audio and MIDI context.
        late=np.zeros(65536,dtype='float32');late[48000:50000]=np.sin(np.arange(2000)*.1)
        self.assertGreater(float((f.mel(late,2)-f.mel(np.zeros_like(late),2)).abs().max()),.1)

    def test_document_conversion_preserves_nonextra_labels(self):
        sys.path[:0]=[str(ROOT/'baselines/scripts'),str(ROOT/'DataCreate/src')]
        import label_datacreate_baselines as old
        import label_datacreate_baselines_v2 as new
        path=PREVIOUS/'predictions/polytune/n00/102/mix.mid';score=PREVIOUS/'bundles/102/verified_score.musicxml'
        a=old.document_from_midi(path,score,model='polytune',sample_id='102')
        b=new.document_from_midi(path,score,model='polytune',sample_id='102')
        self.assertEqual([l for l in a['labels'] if l['type']!='extra_note'],[l for l in b['labels'] if l['type']!='extra_note'])
        self.assertEqual([l['type'] for l in a['labels']],[l['type'] for l in b['labels']])

if __name__=='__main__':unittest.main(verbosity=2)
