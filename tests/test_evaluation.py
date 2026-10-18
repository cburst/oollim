import importlib.util
from pathlib import Path
import tempfile
import unittest
import csv
import json
spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).parents[1] / 'evaluation/statistics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Tests(unittest.TestCase):

    def test_matching(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            i = '1ExampleAb_cdEFGH12345'
            audio = p / ('25025280_' + i + '.m4a.mp3')
            audio.write_bytes(b'audio')
            rows = [{'URL': 'https://drive.google.com/file/d/' + i + '/view?usp=sharing'}]
            (ids, paths) = m.match_files(rows, p)
            self.assertEqual(paths[i], audio.resolve())
            self.assertEqual(m.drive_id('https://drive.google.com/open?id=' + i), i)
            (p / (i + '.wav')).write_bytes(b'other')
            with self.assertRaises(ValueError):
                m.match_files(rows, p)

    def test_statistics(self):
        r = m.measures([10, 20, 30], [10, 20, 30])
        self.assertAlmostEqual(r['Spearman'], 1)
        self.assertEqual(r['QWK'], 1)
        self.assertEqual(m.measures([101, 20], [90, 30])['QWKStatus'], 'OK_clipped')
        raw = m.measures([-5, 20, 101, 110], [0, 30, 90, 100])
        clipped = m.measures([0, 20, 100, 100], [0, 30, 90, 100])
        self.assertAlmostEqual(raw['QWK'], clipped['QWK'])
        self.assertEqual(raw['QWKClippedA'], 3)
        self.assertNotEqual(raw['Spearman'], clipped['Spearman'])
        self.assertEqual(m.measures([10, 10], [10, 10])['QWK'], '')
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for v in [20, 40, 60]:
                row = {f'h{h}-{c}Score': v for h in (1, 2, 3) for c in m.CONSTRUCTS}
                row.update({f'm-{c}Score': v for c in m.CONSTRUCTS})
                row.update({f'speechocean-{c}Score100': v for c in m.CONSTRUCTS})
                rows.append(row)
            m.reports(rows, Path(d), ['speechocean', 'korean'])
            with (Path(d) / 'system_comparison.csv').open(encoding='utf-8-sig') as f:
                r = list(csv.DictReader(f))
            self.assertTrue(all((x['N'] == '0' for x in r if x['Cohort'] == 'all_systems_common')))
            self.assertTrue(all((x['N'] == '3' for x in r if x['Cohort'] == 'available_pair' and x['System'] == 'speechocean')))

class MasterTables(unittest.TestCase):

    def test_matched_points_clusters_and_intervals(self):
        import numpy as np
        rng = np.random.default_rng(5)
        rows = []
        for i in range(40):
            row = {'evaluation_drive_id': str(i // 2)}
            for c in m.CONSTRUCTS:
                for h in (1, 2, 3):
                    row[f'h{h}-{c}Score'] = float(rng.uniform(10, 90))
                row[f'm-{c}Score'] = 120.0 if i == 1 else -5.0 if i == 2 else float(rng.uniform(10, 90))
                row[f'speechocean-{c}Score100'] = float(rng.uniform(10, 90)) if i else ''
            rows.append(row)
        with tempfile.TemporaryDirectory() as d:
            m.master_tables(rows, Path(d), ['speechocean', 'korean'], 100, 42)
            with (Path(d) / 'master_table.csv').open(encoding='utf-8-sig') as f:
                table = list(csv.DictReader(f))
            azure = next((r for r in table if r['Construct'] == 'Accuracy' and r['Method'] == 'azure — three-rater consensus'))
            self.assertEqual(azure['N'], '39')
            self.assertEqual(azure['Speakers'], '20')
            ref = [sum((r[f'h{h}-AccuracyScore'] for h in (1, 2, 3))) / 3 for r in rows[1:]]
            expected = m.measures([r['m-AccuracyScore'] for r in rows[1:]], ref)
            self.assertAlmostEqual(float(azure['QWK']), expected['QWK'])
            self.assertAlmostEqual(float(azure['Spearman']), expected['Spearman'])
            self.assertEqual(azure['QWK_CI_Status'], 'OK')
            self.assertEqual(azure['QWKClippedCount'], '2')
            two = next((r for r in table if r['Construct'] == 'Accuracy' and r['Method'] == 'azure — two-rater consensus'))
            hh = np.array([[r[f'h{j}-AccuracyScore'] for j in (1, 2, 3)] for r in rows[1:]])
            predictions = [r['m-AccuracyScore'] for r in rows[1:]]
            component = [m.measures(predictions, hh[:, [k for k in range(3) if k != j]].mean(1)) for j in range(3)]
            for metric in ('Spearman', 'QWK'):
                self.assertAlmostEqual(float(two[metric]), np.mean([r[metric] for r in component]))
            self.assertEqual(two['N'], azure['N'])
            self.assertEqual(two['QWK_CI_Status'], 'OK')
            baseline = next((r for r in table if r['Construct'] == 'Accuracy' and r['Method'] == 'Human–other-rater consensus agreement'))
            h = np.array([[r[f'h{j}-AccuracyScore'] for j in (1, 2, 3)] for r in rows[1:]])
            expected_baseline = [m.measures(h[:, j], h[:, [k for k in range(3) if k != j]].mean(1)) for j in range(3)]
            for metric in ('Spearman', 'QWK'):
                self.assertAlmostEqual(float(baseline[metric]), np.mean([v[metric] for v in expected_baseline]))
            rendered = (Path(d) / 'master_table.html').read_text()
            self.assertNotIn('<th>Clipped for QWK</th>', rendered)
            self.assertIn('QWK clipping: azure — three-rater consensus: 2', rendered)
            self.assertTrue((Path(d) / 'master_table.html').exists())
            pending = next((r for r in table if r['Method'] == 'korean — three-rater consensus'))
            self.assertEqual(pending['Status'], 'pending_no_scores')
            m.master_tables(rows, Path(d), ['speechocean', 'korean'], 100, 42, available_data=True)
            with (Path(d) / 'master_table_available.csv').open(encoding='utf-8-sig') as f:
                available = list(csv.DictReader(f))
            counts = {r['System'] or r['Method']: r['N'] for r in available if r['Construct'] == 'Accuracy' and r['Reference'] != 'two'}
            self.assertEqual(counts, {'Human–human pairwise agreement': '40', 'Human–other-rater consensus agreement': '40', 'azure': '40', 'speechocean': '39', 'korean': '0'})
if __name__ == '__main__':
    unittest.main()
