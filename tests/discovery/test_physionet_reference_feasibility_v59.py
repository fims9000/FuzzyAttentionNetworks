import unittest
import numpy as np

from scripts.medical.physionet_reference_feasibility_v59 import (
    CHANNELS,INDEX,observed_hours,static_info,aggregate_bin,audit_one_pair
)


class RealPairedMeasurementFeasibilityTests(unittest.TestCase):
    def test_hourly_duplicate_observations_use_actual_median_only(self):
        record=b"Time,Parameter,Value\n00:00,RecordID,123\n00:00,ICUType,2\n00:00,MAP,80\n00:09,MAP,90\n00:02,NIMAP,85\n01:00,NIMAP,-1\n02:00,MAP,72\n"
        self.assertEqual(static_info(record,123),2)
        values,mask=observed_hours(record)
        self.assertEqual(values[0,INDEX["MAP"]],85.)
        self.assertEqual(values[0,INDEX["NIMAP"]],85.)
        self.assertTrue(mask[0,INDEX["MAP"]])
        self.assertFalse(mask[1,INDEX["NIMAP"]])
        self.assertTrue(np.isnan(values[1,INDEX["NIMAP"]]))
        with self.assertRaises(ValueError):
            static_info(record,456)

    def test_static_metadata_scan_does_not_parse_sealed_timeline(self):
        # If static_info scans the later fields this deliberately malformed
        # time-series row raises; frozen split must consume admission only.
        payload=(b"Time,Parameter,Value\n"
                 b"00:00,RecordID,123\n"
                 b"00:00,ICUType,2\n"
                 b"BADLY_MALFORMED_PRIVATE_TIMELINE\n")
        self.assertEqual(static_info(payload,123),2)

    def test_four_hour_aggregation_does_not_fill_unobserved_bins(self):
        x=np.full((4,36,len(CHANNELS)),np.nan,dtype=float)
        m=np.zeros_like(x,dtype=bool)
        x[0,0,INDEX["MAP"]]=75;m[0,0,INDEX["MAP"]]=True
        x[0,3,INDEX["NIMAP"]]=72;m[0,3,INDEX["NIMAP"]]=True
        yy,mm=aggregate_bin(x,m,4)
        self.assertTrue(mm[0,0,INDEX["MAP"]])
        self.assertTrue(mm[0,0,INDEX["NIMAP"]])
        self.assertFalse(mm[0,1,INDEX["MAP"]])
        self.assertTrue(np.isnan(yy[0,1,INDEX["MAP"]]))
        self.assertEqual(yy.shape,(4,9,len(CHANNELS)))

    def test_frozen_overlap_gate_accepts_sufficient_support(self):
        x=np.zeros((360,36,len(CHANNELS)),dtype=float)
        m=np.zeros_like(x,dtype=bool)
        rng=np.random.default_rng(12)
        z=rng.normal(size=(360,36))
        x[:,:,INDEX["MAP"]]=z
        x[:,:,INDEX["NIMAP"]]=z+.02*rng.normal(size=z.shape)
        m[:,:,INDEX["MAP"]]=True
        m[:,:,INDEX["NIMAP"]]=True
        report=audit_one_pair(x,m,"MAP","NIMAP")
        self.assertTrue(report["meets_v57_synthetic_thresholds_for_pair_coverage"])
        self.assertTrue(report["v57_source_alignment_possible_on_measured_association"])
        self.assertTrue(all(v>.9 for v in report["three_disjoint_train_fold_pearson"]))
        self.assertEqual(report["n_patients_at_least_six_aligned_bins"],360)
        self.assertGreater(report["descriptive_observed_only_pearson"],.9)
        self.assertTrue(all(q>=250 for q in report[
            "aligned_pair_cells_in_three_disjoint_train_folds"]))

    def test_unpaired_observed_variables_are_not_usable_reference(self):
        x=np.zeros((360,36,len(CHANNELS)),dtype=float)
        m=np.zeros_like(x,dtype=bool)
        m[:,:18,INDEX["MAP"]]=True
        m[:,18:,INDEX["NIMAP"]]=True
        report=audit_one_pair(x,m,"MAP","NIMAP")
        self.assertEqual(report["both_sensors_present_somewhere_in_history"],360)
        self.assertEqual(report["aligned_joint_time_bins"],0)
        self.assertEqual(report["n_patients_at_least_six_aligned_bins"],0)
        self.assertFalse(report["meets_v57_synthetic_thresholds_for_pair_coverage"])
        self.assertFalse(report["v57_source_alignment_possible_on_measured_association"])
        self.assertIsNone(report["descriptive_observed_only_pearson"])

    def test_invalid_channel_names_and_nondivisible_binning(self):
        x=np.zeros((3,35,len(CHANNELS)))
        with self.assertRaises(ValueError):
            aggregate_bin(x,x>0,4)
        with self.assertRaises(ValueError):
            audit_one_pair(x,x>0,"DOB","MAP")


if __name__=="__main__":
    unittest.main()
