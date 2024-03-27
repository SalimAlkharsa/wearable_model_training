import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import onnx
import onnxruntime as ort
import tensorflow as tf

from onnx_tf.backend import prepare


def convert_onnx_to_tflite(onnx_model_path, tflite_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)

    # Convert ONNX model to TensorFlow model
    tf_rep = prepare(onnx_model)
    graph_def = tf_rep.graph.as_graph_def()

    # Convert TensorFlow model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tf.function(lambda: tf_rep.run(None, None))])
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)


# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Train custom ML model')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, _ = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, 'ei-thesis-flatten-X_training.npy'))
Y_train = np.load(os.path.join(args.data_directory, 'ei-thesis-flatten-y_training.npy'))[:, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=RANDOM_SEED)

print('Training model on', str(X_train.shape[0]), 'inputs...')
# train your model
xgb_classifier = GradientBoostingClassifier(random_state=RANDOM_SEED)
xgb_classifier.fit(X_train, Y_train)
print('Training model OK')
print('')

print('Mean accuracy (training set):', xgb_classifier.score(X_train, Y_train))
print('Mean accuracy (validation set):', xgb_classifier.score(X_test, Y_test))
print('')

print('Converting model...')
# Convert XGBoost model to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(xgb_classifier, initial_types=initial_type)
# Convert ONNX model to TensorFlow Lite
tf_lite_converter = OnnxConverter()
tflite_model = tf_lite_converter.from_onnx_model(onnx_model)

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print('Converting model OK')
print('')