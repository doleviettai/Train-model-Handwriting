import tempfile
import threading
import unittest
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.seq import config_seq as cfg
from src.seq.decode_seq import align_chars, beam_candidates, ctc_log_probability, decode_batch, log_softmax
from src.seq.language_seq import CharacterLanguageModel
from src.seq.metrics_seq import align_edits, edit_distance
from src.seq.model_seq import build_seq_model, ctc_loss, output_lengths
from src.seq.preprocess_seq import process_unit
from src.seq.serve_seq import Recognizer


def perfect_logits(text):
    path = [0]
    for char in text:
        path.extend([cfg.CHAR_TO_ID[char], 0])
    logits = np.full((1, len(path), cfg.N_CLASSES), -15.0, dtype=np.float32)
    logits[0, np.arange(len(path)), path] = 15.0
    return logits


class CTCRegressionTests(unittest.TestCase):
    def test_blank_first_last_character_and_repetitions(self):
        for text in ("ỹ", "aa", "phương", "oôơaâăeê"):
            with self.subTest(text=text):
                logits = perfect_logits(text)
                candidates = beam_candidates(logits, [logits.shape[1]], top_paths=3)
                self.assertIn(text, candidates[0])
                self.assertEqual(decode_batch(logits, [logits.shape[1]], top_paths=3)[0]["text"], text)

    def test_forward_score_matches_tensorflow_loss(self):
        rng = np.random.default_rng(5)
        logits = rng.normal(size=(1, 9, cfg.N_CLASSES)).astype(np.float32)
        for text in ("aa", "ơ", "ăê"):
            labels = tf.constant([[cfg.CHAR_TO_ID[c] for c in text]])
            loss = tf.nn.ctc_loss(labels, logits, [len(text)], [9],
                                  logits_time_major=False, blank_index=0)
            score = ctc_log_probability(log_softmax(logits[0]), text)
            self.assertAlmostEqual(score, -float(loss[0]), places=4)

    def test_no_invented_confidence(self):
        probs = np.zeros((10, cfg.N_CLASSES))
        probs[:, 0] = 1
        self.assertEqual(align_chars("a", probs), [{"c": "a", "p": 0.0}])

    def test_small_accent_keeps_its_bend(self):
        strokes = [np.array([[0, 0], [0, 100]]), np.array([[40, 10], [41, 9], [42, 10]])]
        old, old_start = process_unit(strokes, 0.04, 400, min_stroke_points=2)
        new, new_start = process_unit(strokes, 0.04, 400, min_stroke_points=4)
        old_mark, new_mark = old[np.flatnonzero(old_start)[1]:], new[np.flatnonzero(new_start)[1]:]
        self.assertEqual(len(old_mark), 2)
        self.assertGreaterEqual(len(new_mark), 4)
        self.assertLess(float(new_mark[:, 1].min()), float(new_mark[0, 1]) - 0.001)

    def test_optimal_edit_alignment_accounts_for_all_errors(self):
        for ref, hyp in (("phương", "Phưng"), ("ô", "o"), ("aa", "a"), ("", "a")):
            pairs = align_edits(ref, hyp)
            self.assertEqual("".join(a for a, _ in pairs), ref)
            self.assertEqual("".join(b for _, b in pairs), hyp)
            self.assertEqual(sum(a != b for a, b in pairs), edit_distance(ref, hyp))

    def test_language_model_roundtrip_and_unknown_word(self):
        lm = CharacterLanguageModel.fit(["phương", "phương", "tôi", "là"])
        self.assertNotIn("bí_mật_validation", lm.words)
        self.assertIn("phương", lm.nearby_words("phưng"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lm.json"
            lm.save(path)
            loaded = CharacterLanguageModel.load(path)
            self.assertEqual(lm.log_probability("phương"), loaded.log_probability("phương"))

    def test_expected_answer_does_not_change_recognition(self):
        class FakeModel:
            def __call__(self, inputs, training=False):
                n = int(output_lengths(inputs[1]).numpy()[0])
                logits = np.full((1, n, cfg.N_CLASSES), -15, dtype=np.float32)
                logits[:, :, 0] = 15
                logits[0, n // 2, 0] = -15
                logits[0, n // 2, cfg.CHAR_TO_ID["o"]] = 15
                return tf.constant(logits)

        recognizer = Recognizer.__new__(Recognizer)
        recognizer.model = FakeModel()
        recognizer.preprocessing = dict(step=0.04, max_t=400, min_stroke_points=4)
        recognizer.language_model = CharacterLanguageModel.fit(["o", "ô", "ơ"])
        recognizer.lock = threading.Lock()
        strokes = [[[0, 0], [5, 10], [10, 0]]]
        a = recognizer.predict(strokes, expected="o")
        b = recognizer.predict(strokes, expected="ô")
        self.assertEqual(a, b)
        self.assertEqual(a["text"], "o")


class ModelStabilityTests(unittest.TestCase):
    def test_predictions_ignore_padding_and_save_load(self):
        keras.utils.set_random_seed(11)
        model = build_seq_model(np.arange(5) / 3, np.ones(5) * 2, units=8, dropout=0)
        features = np.random.default_rng(1).normal(size=(1, 13, 5)).astype(np.float32)
        lengths = tf.constant([13])
        reference = model([tf.constant(features), lengths], training=False).numpy()
        for total in (14, 31, 32):
            padded = np.pad(features, ((0, 0), (0, total - 13), (0, 0)))
            actual = model([tf.constant(padded), lengths], training=False).numpy()
            np.testing.assert_allclose(actual[:, :7], reference, atol=2e-5, rtol=2e-5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            model.save(path)
            loaded = keras.models.load_model(path)
            np.testing.assert_allclose(loaded([tf.constant(features), lengths]).numpy(), reference, atol=1e-6)
        labels = tf.constant([[cfg.CHAR_TO_ID["ô"]]])
        loss = ctc_loss(tf.constant(reference), lengths, labels, tf.constant([1]))
        self.assertTrue(np.isfinite(loss.numpy()).all())


if __name__ == "__main__":
    unittest.main()
