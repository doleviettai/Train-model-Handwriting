import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import keras
import numpy as np

from src.char.common import write_json
from src.ghep import data_ghep as data
from src.ghep.features_ghep import N_GEOMETRY, geometry
from src.ghep.inference_ghep import Predictor
from src.ghep.hybrid_ghep import select_weight
from src.ghep.model_ghep import build_model, initialize
from src.ghep.train_ghep import _atomic_save
from src.ghep.collect_ghep import save_correction


def write_ink(path, truth, split, style, namespace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = ' xmlns="http://www.w3.org/2003/InkML"' if namespace else ''
    path.write_text(f'''<ink{extra}><annotationXML><Data_Source>synthetic</Data_Source>
<Dataset_Split>{split}</Dataset_Split><Style_ID>{style}</Style_ID><Writer_ID>{style}</Writer_ID>
<Template_Family>test</Template_Family></annotationXML><traceGroup><annotationXML>
<Tg_Truth>{truth}</Tg_Truth></annotationXML><trace>0 0,1 2,2 0,3 1</trace></traceGroup></ink>''', encoding="utf-8")


class GhepStabilityTests(unittest.TestCase):
    def test_correction_requires_explicit_valid_label_and_writer(self):
        strokes = [[[0, 0], [1, 2], [2, 1]]]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for label, writer in (("", "tai01"), ("outside", "tai01"), ("th", "")):
                with self.assertRaises(ValueError):
                    save_correction(strokes, label, writer, ["th", "anh"], raw)
            self.assertEqual(list(raw.iterdir()), [])
            result = save_correction(strokes, "th", "tai<&01", ["th", "anh"], raw)
            item = data.parse_inkml(raw / "browser_corrections" / result["file"])
            self.assertEqual(item["truth"], "th")
            self.assertEqual(item["writer"], "tai<&01")
            self.assertEqual(item["split"], "train")
            self.assertEqual(item["source"], "human")
            np.testing.assert_array_equal(item["strokes"][0], strokes[0])

    def test_balanced_epoch_does_not_favor_the_frequent_class(self):
        examples = [{"strokes": [np.array([[0, 0], [1, 2], [2, 0]], np.float32)], "y": label}
                    for label in [0] + [1] * 20]
        dataset = data.make_dataset(examples, 4, True, samples_per_class=3)
        labels = np.concatenate([y.numpy() for _, y in dataset])
        self.assertEqual(np.bincount(labels).tolist(), [3, 3])

    def test_composition_uses_only_training_glyphs_and_marks_derivation(self):
        import pandas as pd
        from types import SimpleNamespace
        from src.ghep.compose_ghep import compose_samples, source_files
        metadata = pd.DataFrame({"sample_id": [0, 1, 2], "path": ["train_t", "test_t", "train_h"], "truth": ["t", "t", "h"]})
        splits = metadata.assign(split=["train", "test", "train"])
        def fake_read(path, **kwargs):
            return metadata if path.name == "samples_meta.csv" else splits
        with patch('src.ghep.compose_ghep.pd.read_csv', side_effect=fake_read):
            selected = source_files()
        self.assertEqual(selected['path'].tolist(), ['train_t', 'train_h'])
        def glyph(path):
            self.assertNotEqual(path, "test_t")
            return SimpleNamespace(truth="t" if path == "train_t" else "h", strokes=[], canvas_w=1, canvas_h=1)
        with patch('src.ghep.compose_ghep.source_files', return_value=selected), \
             patch('src.ghep.compose_ghep.parse_inkml', side_effect=glyph), \
             patch('src.ghep.compose_ghep.normalize_to_canvas', return_value=[np.array([[0,0],[1,1]], np.float32)]):
            items = compose_samples(["th"], 3)
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual(item['source'], 'composed')
            self.assertEqual(item['split'], 'train')
            self.assertEqual(item['provenance'], ['train_t','train_h'])

    def test_hybrid_rejects_improvement_that_hurts_ue_ie_or_one_source(self):
        baseline = {"alpha": 0, "score": 0.7, "pair_correct": 2, "source_accuracy": {"human": 0.5, "synthetic": 0.9}}
        pair_worse = {"alpha": 0.15, "score": 0.9, "pair_correct": 1, "source_accuracy": {"human": 0.9, "synthetic": 0.9}}
        source_worse = {"alpha": 0.3, "score": 0.9, "pair_correct": 2, "source_accuracy": {"human": 1.0, "synthetic": 0.8}}
        self.assertEqual(select_weight([baseline, pair_worse, source_worse])["alpha"], 0)
        safe = {"alpha": 0.5, "score": 0.8, "pair_correct": 2, "source_accuracy": {"human": 0.7, "synthetic": 0.9}}
        self.assertEqual(select_weight([baseline, pair_worse, source_worse, safe])["alpha"], 0.5)

    def test_namespaced_inkml_and_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "sample.inkml"
            write_ink(p, "uê", "train", "s1", namespace=True)
            item = data.parse_inkml(p)
            self.assertEqual(item["truth"], "uê")
            self.assertEqual(item["writer"], "s1")
        for strokes in ([], [[[0, float('nan')]]], [[1, 2, 3]], [[[0, 1, 2]]]):
            with self.subTest(strokes=strokes), self.assertRaises(ValueError):
                data.normalize(strokes)

    def test_cache_refreshes_and_repairs_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, cache = root / "raw", root / "cache"
            for split, style in (("train", "s1"), ("validation", "s2"), ("test", "s3")):
                write_ink(raw / "one" / f"{split}.inkml", "uê", split, style)
                # Không để bản sao hình học của cùng mẫu rò rỉ giữa các split.
                p = raw / "one" / f"{split}.inkml"
                text = p.read_text(encoding="utf-8")
                text = text.replace("3 1", f"3 {1 + ('train','validation','test').index(split)}")
                p.write_text(text, encoding="utf-8")
            first = data.build_processed(folder=cache, raw_dir=raw)
            write_ink(raw / "one" / "extra.inkml", "uê", "train", "s1")
            second = data.build_processed(folder=cache, raw_dir=raw)
            self.assertNotEqual(first["source_sha256"], second["source_sha256"])
            self.assertEqual(len(data.load_split("train", cache)), 2)
            (cache / "train.pkl").unlink()
            data.build_processed(folder=cache, raw_dir=raw)
            self.assertEqual(len(data.load_split("train", cache)), 2)
            data.verify_snapshot(cache)
            (cache / "label_map.json").write_text('{}')
            with self.assertRaises(ValueError):
                data.verify_snapshot(cache)

    def test_writer_leakage_is_rejected(self):
        examples = [{"strokes": [np.array([[0, 0], [i + 1, 1]], np.float32)], "truth": "uê", "split": split,
                     "writer": "same_writer", "source": "human", "file": str(i)}
                    for i, split in enumerate(("train", "validation", "test"))]
        with self.assertRaisesRegex(ValueError, "trùng"):
            data._validate(examples, ["uê"])

    def test_derived_validation_rejects_shared_training_glyph(self):
        examples = [{"strokes": [np.array([[0, 0], [i + 1, 1]], np.float32)], "truth": "th", "split": split,
                     "writer": "derived_same_writer", "source": "composed", "file": str(i),
                     "provenance": ["training_glyph_t", f"glyph_h_{i}"]}
                    for i, split in enumerate(("train", "validation", "test"))]
        with self.assertRaisesRegex(ValueError, "Glyph nguồn trùng"):
            data._validate(examples, ["th"])

    def test_geometry_sees_direction_and_dot_that_image_alone_cannot_order(self):
        strokes = [np.array([[0, 0], [0, 1], [0.3, 1], [0.3, 0]], np.float32)]
        forward, backward = geometry(strokes), geometry([strokes[0][::-1]])
        self.assertEqual(forward.shape, (N_GEOMETRY,))
        self.assertGreater(float(np.linalg.norm(forward - backward)), 0.1)
        with_dot = geometry([*strokes, np.array([[0.1, -0.2]], np.float32)])
        self.assertGreater(float(np.linalg.norm(forward - with_dot)), 0.1)
        np.testing.assert_allclose(geometry([s * 3 + 5 for s in strokes]), forward, atol=1e-5)
        self.assertTrue(np.isfinite(geometry([np.array([[0, 0]])])).all())

    def test_transfer_starts_with_legacy_predictions(self):
        keras.utils.set_random_seed(5)
        source = build_model(3, dropout=0)
        target = build_model(3, mean=np.zeros(N_GEOMETRY), variance=np.ones(N_GEOMETRY), dropout=0)
        initialize(target, source)
        x = np.random.default_rng(2).random((1, 64, 192, 1)).astype(np.float32)
        actual = target({"image": x, "geometry": np.zeros((1, N_GEOMETRY), np.float32)}, training=False)
        np.testing.assert_allclose(actual.numpy(), source(x, training=False).numpy(), atol=1e-6)

    def test_image_transfer_from_combined_model_and_geometry_outlier_bound(self):
        source = build_model(3, mean=np.zeros(N_GEOMETRY), variance=np.ones(N_GEOMETRY), dropout=0)
        target = build_model(3, mean=np.zeros(N_GEOMETRY), variance=np.ones(N_GEOMETRY), dropout=0,
                             robust_geometry=True)
        initialize(target, source)
        kernel = target.get_layer('classifier_dense').get_weights()[0]
        np.testing.assert_array_equal(kernel[-32:], 0)
        bound = keras.Model(target.inputs, target.get_layer('geometry_bound').output)
        result = bound({'image':np.zeros((1,64,192,1),np.float32),
                        'geometry':np.full((1,N_GEOMETRY),1e8,np.float32)})
        self.assertLessEqual(float(np.max(np.abs(result.numpy()))), 1.0)

    def test_checkpoint_contains_optimizer_lr_and_patience_in_one_file(self):
        model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(2, activation="softmax")])
        model.compile(optimizer=keras.optimizers.AdamW(0.001), loss="sparse_categorical_crossentropy")
        model.train_on_batch(np.ones((2, 1)), np.array([0, 1]))
        model.optimizer.learning_rate.assign(0.0005)
        bundle = {"state": {"epoch": 4, "plateau": 2, "bad_epochs": 3}, "history": [{"epoch": 4}]}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "ghep_last.keras"
            _atomic_save(model, p, bundle)
            with zipfile.ZipFile(p) as archive:
                self.assertEqual(json.loads(archive.read("ghep_resume.json")), bundle)
            loaded = keras.models.load_model(p)
            self.assertEqual(int(loaded.optimizer.iterations), 1)
            self.assertAlmostEqual(float(loaded.optimizer.learning_rate.numpy()), 0.0005)

    def test_inference_uses_run_labels_even_when_global_cache_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run / "label_map.json", {"classes": ["uê", "iê"]})
            model = keras.Sequential([keras.Input((64, 192, 1)), keras.layers.GlobalAveragePooling2D(),
                                      keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros", activation="softmax")])
            model.save(run / "ghep_best.keras")
            with patch.object(data, "PROC_DIR", run / "nonexistent_global_cache"):
                predictor = Predictor(run)
                self.assertEqual(predictor.classes, ["uê", "iê"])
                probs = predictor.probabilities([[np.array([[0, 0], [1, 1]])]])
                self.assertEqual(probs.shape, (1, 2))
                (run / "data").mkdir()
                (run / "label_map.json").unlink()
                with self.assertRaisesRegex(ValueError, "thiếu bảng nhãn"):
                    Predictor(run)


if __name__ == "__main__":
    unittest.main()
