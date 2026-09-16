"""Egitim ciktilarini pahali benchmark baslamadan once dogrular."""

import argparse
import gc
import hashlib
import os
from pathlib import Path

# Bu ortamda HDF5 kutuphanesinin TensorFlow'dan once yuklenmesi daha guvenlidir.
import h5py  # noqa: F401

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


MODEL_SUFFIXES = {".h5", ".hdf5", ".keras"}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def activation_name(layer):
    activation = getattr(layer, "activation", None)
    if activation is None:
        return None
    name = getattr(activation, "__name__", None)
    if name:
        return str(name).lower()
    return str(tf.keras.activations.serialize(activation)).lower()


def validate_model_file(path, expected_output_activation):
    model = tf.keras.models.load_model(str(path), compile=False)
    try:
        if len(model.outputs) != 1:
            raise ValueError(f"tek cikis bekleniyordu; bulunan={len(model.outputs)}")
        if not model.layers:
            raise ValueError("model katmani bulunamadi")

        actual_activation = activation_name(model.layers[-1])
        if actual_activation != expected_output_activation:
            raise ValueError(
                "son katman aktivasyonu uyusmuyor: "
                f"beklenen={expected_output_activation}, bulunan={actual_activation}"
            )

        output_shape = model.output_shape
        if not isinstance(output_shape, tuple) or len(output_shape) != 4:
            raise ValueError(f"goruntu cikisi (N,H,W,C) bekleniyordu: {output_shape}")

        return actual_activation, output_shape
    finally:
        del model
        tf.keras.backend.clear_session()
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-output-activation",
        default="tanh",
        # sigmoid desteklenir: top_modeller/ altindaki 0.68-0.70 success_25m
        # alan modellerin hepsi sigmoid ciktiliydi ve benchmark cikarim yolu
        # sigmoid'i zaten dogru olcekliyor (goruntu_islemleri.py
        # prediction_to_uint8 -> v * 255, model_activation_output_mode).
        choices=("tanh", "sigmoid"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model dizini bulunamadi: {model_dir}")

    model_files = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )
    if not model_files:
        raise SystemExit(f"Dogrulanacak model bulunamadi: {model_dir}")

    unique_by_hash = {}
    for path in model_files:
        unique_by_hash.setdefault(sha256_file(path), path)

    print(
        f"Model on kontrolu: dosya={len(model_files)}, "
        f"benzersiz_icerik={len(unique_by_hash)}"
    )
    for path in unique_by_hash.values():
        try:
            activation, output_shape = validate_model_file(
                path,
                args.expected_output_activation,
            )
        except Exception as exc:
            raise SystemExit(f"Gecersiz model: {path.name} | {exc}") from exc
        print(
            f"  OK | {path.name} | cikis_aktivasyonu={activation} | "
            f"cikis_boyutu={output_shape}"
        )

    print("Tum benzersiz modeller cikis sozlesmesini sagliyor.")


if __name__ == "__main__":
    main()
