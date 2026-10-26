"""Exactness checks using synthetic scores only; no private input is required."""
import unittest
import numpy as np
from scipy.stats import rankdata
from evaluation.paired_four_ridges import WeightedMetrics, SYSTEMS, COMPARISONS

def explicit(a,b):
    rho=np.corrcoef(rankdata(a),rankdata(b))[0,1]
    a=np.floor(np.clip(a,0,100)+.5);b=np.floor(np.clip(b,0,100)+.5)
    den=np.mean(a*a)+np.mean(b*b)-2*a.mean()*b.mean()
    return np.array([rho,1-np.mean((a-b)**2)/den])

class PairedBootstrapTests(unittest.TestCase):
    def test_weighted_equals_explicit_resampling(self):
        rng=np.random.default_rng(78);m=len(SYSTEMS);n=35
        values=rng.integers(-10,115,(m+6,n)).astype(float)
        h=values[m:m+3].copy()
        values[m+3:]=[np.delete(h,j,axis=0).mean(0) for j in range(3)]
        engine=WeightedMetrics(values)
        for _ in range(25):
            idx=rng.integers(0,n,n);counts=np.bincount(idx,minlength=n)[None]
            machine,human=engine.evaluate(counts)
            for i in range(m):
                expected=np.mean([explicit(values[i,idx],values[m+3+j,idx]) for j in range(3)],axis=0)
                np.testing.assert_allclose(machine[0,i],expected,atol=1e-12)
            expected=np.mean([explicit(values[m+j,idx],values[m+3+j,idx]) for j in range(3)],axis=0)
            np.testing.assert_allclose(human[0],expected,atol=1e-12)
    def test_constant_scores_remain_undefined(self):
        machine,human=WeightedMetrics(np.ones((len(SYSTEMS)+6,12))).evaluate(np.ones((1,12)))
        self.assertTrue(np.isnan(machine).all());self.assertTrue(np.isnan(human).all())
    def test_full_comparison_family(self):
        self.assertEqual(len(SYSTEMS),36);self.assertEqual(COMPARISONS,510)
if __name__=='__main__':unittest.main()


def test_contrast_plan_has_correct_references():
    from evaluation.paired_four_ridges import SYSTEMS, CONSTRUCTS, PREVIOUS_LEADERS
    baselines={(c,s,r,m) for c in CONSTRUCTS for s in SYSTEMS for r in ('human_other_two','azure_two_rater') for m in ('rho','qwk') if not (s=='azure' and r=='azure_two_rater')}
    previous={(c,s,PREVIOUS_LEADERS[c,m],m) for c in CONSTRUCTS for s in SYSTEMS if s.startswith('banded_3ratio_') for m in ('rho','qwk')}
    assert len(baselines)==426 and len(previous)==84
    assert len(baselines|previous)==510
    assert all(r in SYSTEMS and not r.startswith('banded_3ratio_') for c,s,r,m in previous)
