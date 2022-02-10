# Reference: https://keras.io/examples/vision/video_classification/
import os.path as op
import numpy as np
from tensorflow import keras
import cv2


IMG_SIZE = 224  # default image size
MAX_SEQ_LENGTH = 120  # RNN sequence length
NUM_FEATURES = 2048  # number of feature for each frame after inception v3
label_processor = ["deadlift", "others", "squat"]


def get_sequence_model(model_path=None):
    class_vocab = label_processor

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    if model_path is not None:
        rnn_model.load_weights(model_path)
    return rnn_model


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def crop_center_square(frame: np.ndarray):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(
    path: str,
    max_frames: int = 0,
    resize: tuple = (IMG_SIZE, IMG_SIZE),
    fsample: int = 5,
):
    """
    Load video from a given path, down sample with the rate
    of `fsample` and crop the center square of each frame.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    if len(frames) > fsample:
        return np.array(frames)[::fsample, :, :, :]  # downsample
    else:
        return np.array(frames)


def prepare_single_video(frames: np.ndarray, feature_extractor):
    """Prepare frame features and mask for sequence prediction"""
    frames = frames[None, ...]
    frame_mask = np.zeros(
        shape=(
            1,
            MAX_SEQ_LENGTH,
        ),
        dtype="bool",
    )
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path, sequence_model, feature_extractor, verbose: bool = True):
    """
    Predict exercise type of a given path to video
    Usage
    =====
    >>> sequence_model = get_sequence_model("MODEL_PATH")
    >>> probabilities = sequence_prediction("VIDEO_PATH", sequence_model)
    """
    if not op.exists(path):
        print("Video path does not exists...")
        return
    class_vocab = label_processor
    frames = load_video(path)
    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    probabilities_dict = {}
    for i in np.argsort(probabilities)[::-1]:
        probabilities_dict[class_vocab[i]] = probabilities[i]
        if verbose:
            print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames, probabilities_dict
