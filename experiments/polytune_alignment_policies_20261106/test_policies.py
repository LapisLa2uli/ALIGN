import unittest
from policy_lib import TimeScoreLookup, timestamp_score, note_score

def label(a, b, kind='wrong_note', indices=None):
    return dict(type=kind, start_time=a, end_time=b, score_event_indices=indices or [])

def lookup(rows):
    return TimeScoreLookup(dict(notes=[dict(start=a, end=b, score_event_indices=idx, ordinal=i, pitch=60)
                                     for i,(a,b,idx) in enumerate(rows)],
                                score=[dict(ql_start=i,ql_end=i+1,pitch=60) for i in range(10)]))

class Policies(unittest.TestCase):
    def test_timestamp_duplicate_and_type(self):
        g=[label(1,2)]
        self.assertEqual(timestamp_score(g,[label(1,2),label(1,2)])['credit'],1)
        self.assertEqual(timestamp_score(g,[label(1,2,'missed_note')])['credit'],0)

    def test_overlap_and_onset_are_distinct(self):
        g=[label(1,2)];p=[label(1,10)]
        self.assertEqual(timestamp_score(g,p)['credit'],0)
        self.assertEqual(timestamp_score(g,p,'onset',.1)['credit'],1)

    def test_exclusive_assignment_maximizes_matches(self):
        g=[label(0,1),label(.14,1.14)]
        p=[label(.05,1.05),label(-.09,.91)]
        self.assertEqual(timestamp_score(g,p,'onset',.1)['credit'],2)

    def test_adjacent_notes_and_repeated_passage(self):
        l=lookup([(1,2,[0]),(2,3,[1]),(4,5,[0]),(5,6,[1])])
        p,_=l.project([label(2.02,2.9),label(4.01,4.9)])
        self.assertEqual([x['score_event_indices'] for x in p],[[1],[0]])

    def test_no_mapping_in_silence_or_unaligned_note(self):
        p,_=lookup([(1,2,[0]),(2,3,[])]).project([label(0,.5),label(2,3)])
        self.assertEqual([x['score_event_indices'] for x in p],[[],[]])

    def test_missing_note_gap_interpolation(self):
        p,t=lookup([(1,2,[0]),(3,4,[2])]).project([label(2,3,'missed_note')])
        self.assertEqual(p[0]['score_event_indices'],[1])
        self.assertEqual(t[0]['method'],'interpolated_score_gap')

    def test_extras_and_backward_jump(self):
        p,_=lookup([(1,2,[0]),(3,4,[1])]).project([label(2.1,2.8,'extra_note')])
        self.assertEqual(p[0]['score_event_indices'],[0,1])
        p,_=lookup([(1,2,[4]),(3,4,[0])]).project([label(2.1,2.8,'extra_note')])
        self.assertEqual(p[0]['score_event_indices'],[])

    def test_preserve_error_type_count_copy_and_clear_old_location(self):
        x=label(0,.5,'repetition',[9]);x.update(extra_copies=2,score_part={'start_note_index':9,'end_note_index':9})
        p,_=lookup([(1,2,[0])]).project([x])
        self.assertEqual(len(p),1);self.assertEqual(p[0]['type'],'repetition')
        self.assertEqual(p[0]['extra_copies'],2);self.assertEqual(p[0]['score_event_indices'],[])
        self.assertNotIn('score_part',p[0])

    def test_note_grouping_and_duplicate_penalty(self):
        g=[label(0,1,indices=[0,1])]
        p=[label(0,.5,indices=[0]),label(.5,1,indices=[1])]
        self.assertEqual(note_score(g,p)['f1'],1)
        self.assertLess(note_score(g,p+p)['f1'],1)

if __name__=='__main__':
    unittest.main(verbosity=2)
