import json
import tempfile
import unittest
from pathlib import Path
from scripts.experiment_ioi_repeated_draws_v31 import (
    DRAW_CONFIGS, NN_SEEDS, aggregate
)

class NewDrawReplicationContracts(unittest.TestCase):
    def test_two_new_draws_differ_from_v27_reference(self):
        self.assertEqual(len(DRAW_CONFIGS),2)
        self.assertEqual(len({x[0] for x in DRAW_CONFIGS}),2)
        self.assertNotIn(20261107,[x[0] for x in DRAW_CONFIGS])
        self.assertEqual(NN_SEEDS,(11,23,37))

    def test_aggregate_does_not_count_artifact_as_independent(self):
        draws=[]
        for i,(draw_seed,_,_) in enumerate(DRAW_CONFIGS):
            rows=[]
            for world in ("complete","omitted","artifact"):
                for arm in ("joint","staged"):
                    for seed in NN_SEEDS:
                        rows.append({
                           "family":world,"method":arm,"seed":seed,
                           "incremental_learned_concept_nll_gain":(.02 if arm=="joint" else 0.),
                           "paired_test_subject_bootstrap_ci95":[.01,.03] if arm=="joint" else [-.01,.01],
                           "max_abs_hidden_factor_correlation":.2
                        })
            draws.append((draw_seed,{
                "generator_seed":draw_seed,"n_training_runs":12,"leakage_audit":rows}))
        with tempfile.TemporaryDirectory() as t:
            result=aggregate(Path(t),draws)
            self.assertEqual(result["n_distinct_test_draws"],2)
            self.assertEqual(result["n_fitted_training_runs"],24)
            self.assertEqual(len(result["per_fit"]),24)
            self.assertEqual(result["outcomes"]["omitted:joint"]["positive_gains"],6)
            self.assertTrue((Path(t)/"V31_SUMMARY.json").exists())

if __name__=="__main__":
    unittest.main()
