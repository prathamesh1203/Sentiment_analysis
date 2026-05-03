"""
Run this script ONCE locally (requires TensorFlow installed):
    python convert_to_onnx.py

It will produce:  models/lstm_model.onnx
Commit that file to git, then the app uses onnxruntime (no TF needed on cloud).
"""
import os
import sys
import subprocess
import tensorflow as tf

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH       = os.path.join(BASE_DIR, "models", "lstm_model.keras")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_saved_model")
ONNX_PATH        = os.path.join(BASE_DIR, "models", "lstm_model.onnx")

# ── Step 1: Load .keras and re-save as SavedModel (tf2onnx CLI handles this best)
print(f"Loading model from: {KERAS_PATH}")
model = tf.keras.models.load_model(KERAS_PATH)

print(f"Re-saving as SavedModel to: {SAVED_MODEL_PATH}")
model.export(SAVED_MODEL_PATH)          # tf.keras v3 export → SavedModel
print("SavedModel saved ✓")

# ── Step 2: Convert SavedModel → ONNX via tf2onnx CLI
print("\nConverting SavedModel → ONNX (opset 13) …")
result = subprocess.run(
    [sys.executable, "-m", "tf2onnx.convert",
     "--saved-model", SAVED_MODEL_PATH,
     "--output",      ONNX_PATH,
     "--opset",       "13"],
    capture_output=True, text=True
)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:\n", result.stderr)
    raise RuntimeError("tf2onnx conversion failed — see error above")

size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"\n✅  ONNX model saved to: {ONNX_PATH}  ({size_mb:.1f} MB)")
print("    Now commit  models/lstm_model.onnx  and push to GitHub.")
