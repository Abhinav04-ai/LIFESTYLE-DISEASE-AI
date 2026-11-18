import tempfile
import unittest
from pathlib import Path

from src.data_loader import DatasetError, load_dataset
from src.paper_reference import PAPER_CV_METRICS, PAPER_FIGURES, PAPER_TEST_METRICS
from src.prediction import validate_input


class DatasetValidationTests(unittest.TestCase):
    def test_missing_dataset_reports_expected_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DatasetError, "pima-indians-diabetes.data.txt"):
                load_dataset("diabetes", Path(directory))


class PredictionInputTests(unittest.TestCase):
    def test_diabetes_input_is_ordered_to_expected_schema(self):
        values = {
            "Pregnancies": 2,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 25.5,
            "DiabetesPedigreeFunction": 0.4,
            "Age": 35,
        }
        frame = validate_input("diabetes", values)
        self.assertEqual(list(frame.columns), list(values))
        self.assertEqual(frame.shape, (1, 8))

    def test_heart_input_rejects_unknown_category(self):
        values = {
            "age": 54,
            "sex": 1,
            "cp": 5,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        }
        with self.assertRaisesRegex(ValueError, "cp must be one"):
            validate_input("heart", values)


class PaperReferenceTests(unittest.TestCase):
    def test_paper_figures_and_reference_metrics_are_available(self):
        self.assertTrue(all(path.is_file() for path in PAPER_FIGURES.values()))
        self.assertEqual(PAPER_CV_METRICS["diabetes"]["rf"]["mean_f1"], 0.6705)
        self.assertEqual(PAPER_CV_METRICS["heart"]["rf"]["mean_f1"], 0.8185)
        self.assertEqual(PAPER_TEST_METRICS["heart"]["xgb"]["F1-score"], 0.9)


if __name__ == "__main__":
    unittest.main()
