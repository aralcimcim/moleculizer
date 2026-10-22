import math
import unittest

from workbench import describe, evidence_points, validate_pins


class WorkbenchTests(unittest.TestCase):
    def test_calculated_descriptors_and_structure_image(self):
        result = describe("CCO")
        self.assertAlmostEqual(result["weight"], 46.07, places=1)
        self.assertEqual(result["donors"], 1)
        self.assertEqual(result["acceptors"], 1)
        self.assertEqual(result["rings"], 0)
        self.assertTrue(math.isfinite(result["logp"]))
        self.assertTrue(result["image"].startswith("data:image/svg+xml;base64,"))

    def test_points_preserve_source_values_labels_and_units(self):
        record = dict(source_id="esol:row:2", smiles="CCO", source_url="https://example.org",
                      text="Measured solubility", standard_type="Measured log10 aqueous solubility",
                      standard_value="-1.5", standard_units="log10(mol/L)", labels={"NR-AR": None})
        point = evidence_points([record])[0]
        self.assertEqual(point["value"], -1.5)
        self.assertEqual(point["units"], "log10(mol/L)")
        self.assertIsNone(point["labels"]["NR-AR"])
        self.assertEqual(point["id"], "esol:row:2")
        self.assertEqual(point["url"], record["source_url"])

    def test_pins_are_bounded_unique_and_source_restricted(self):
        self.assertEqual(validate_pins(["a", "b"], {"a", "b"}), ["a", "b"])
        for value in (["unknown"], ["a", "a"], ["a", "b", "c", "d"], "a"):
            with self.assertRaises(ValueError):
                validate_pins(value, {"a", "b", "c", "d"})

    def test_point_payload_is_bounded(self):
        records = [dict(activity_id=i, smiles="CCO", text="Source", source_url="https://example.org") for i in range(130)]
        self.assertEqual(len(evidence_points(records)), 120)


if __name__ == "__main__":
    unittest.main()
