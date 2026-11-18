import tempfile
import unittest
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.char.features import deep_features, geometry_features
from src.char.hybrid import CHARS, PARTS, blend, valid_probabilities
from src.char.model import build_model, initialize_encoder
from src.char.train import save_model
from src.labels import BASES, MODIFIERS, TONES


class CharRegressionTests(unittest.TestCase):
    def test_transfer_preserves_existing_classifier_before_geometry_learning(self):
        from src.model import build_model as legacy_model
        norm = keras.layers.Normalization(mean=np.zeros(7), variance=np.ones(7), name="feature_norm")
        source = legacy_model(40, 7, norm)
        target = build_model(40, np.zeros(7), np.ones(7))
        initialize_encoder(target, source)
        x = np.random.default_rng(9).normal(size=(2, 40, 7)).astype(np.float32)
        before, after = source(x, training=False), target(x, training=False)
        for head in before:
            np.testing.assert_allclose(after[head].numpy(), before[head].numpy(), atol=1e-6)

    def test_penup_jump_is_not_a_written_stroke(self):
        p = np.array([[[0, 0], [0.1, 0.1], [0.8, 0.8], [0.9, 0.9]]], np.float32)
        s = np.array([[1, 0, 1, 0]], np.float32)
        f = deep_features(p, s).numpy()
        np.testing.assert_array_equal(f[0, 2, 4:6], [0, 0])
        self.assertGreater(float(np.linalg.norm(f[0, 1, 4:6])), 0)

    def test_geometry_records_ascender_and_is_finite_for_dot(self):
        angle = np.linspace(0, 2 * np.pi, 32)
        a = np.stack([0.4 + 0.15 * np.cos(angle), 0.65 + 0.15 * np.sin(angle)], axis=1)
        d = a.copy()
        d[-5:, 0] = 0.55
        d[-5:, 1] = np.linspace(0.8, 0.2, 5)
        p = np.array([a, d, np.ones_like(a) * 0.5], np.float32)
        s = np.zeros(p.shape[:2], np.float32)
        s[:, 0] = 1
        g = geometry_features(p, s)
        self.assertTrue(np.isfinite(g).all())
        self.assertGreater(float(np.linalg.norm(g[0] - g[1])), 0.1)

    def test_valid_character_projection_and_disabled_svm(self):
        heads = [np.ones((2, n)) / n for n in (len(BASES), len(MODIFIERS), len(TONES))]
        shallow = np.zeros_like(heads[0])
        shallow[:, BASES.index("d")] = 1
        np.testing.assert_array_equal(blend(heads, shallow, 0)[0], heads[0])
        probabilities = valid_probabilities(blend(heads, shallow, 1))
        np.testing.assert_allclose(probabilities.sum(1), 1)
        self.assertTrue(np.all(np.asarray(CHARS)[probabilities.argmax(1)] == "d"))
        self.assertEqual(len(set(CHARS)), 196)

    def test_model_checkpoint_preserves_optimizer_and_state(self):
        import json
        import zipfile
        model = build_model(40, np.zeros(7), np.ones(7), dropout=0.1)
        model.compile(optimizer=keras.optimizers.AdamW(0.0003),
                      loss={h: "sparse_categorical_crossentropy" for h in ("base", "modifier", "tone")})
        x = np.random.default_rng(4).normal(size=(2, 40, 7)).astype(np.float32)
        y = {h: np.zeros(2, np.int32) for h in ("base", "modifier", "tone")}
        model.train_on_batch(x, y)
        before = model(x, training=False)
        bundle = {"state": {"completed_epochs": 1}, "history": [{"epoch": 1}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "last.keras"
            save_model(model, path, bundle)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(json.loads(archive.read("char_resume.json")), bundle)
            loaded = keras.models.load_model(path)
            self.assertEqual(int(loaded.optimizer.iterations.numpy()), 1)
            for key, values in before.items():
                np.testing.assert_allclose(loaded(x, training=False)[key].numpy(), values.numpy(), atol=1e-6)
            loaded.train_on_batch(x, y)
            self.assertEqual(int(loaded.optimizer.iterations.numpy()), 2)


if __name__ == "__main__":
    unittest.main()
