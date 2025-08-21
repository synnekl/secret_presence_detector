import os, random, pathlib
import numpy as np
import tensorflow as tf


# Link to dataset https://www.kaggle.com/datasets/antfilatov/mini-speech-commands

# -----------------------
# Config
# -----------------------
DATA_DIR = pathlib.Path("mini_speech_commands")
TARGET = "secret"           # code word to detect
SAMPLE_RATE = 16000
DURATION_S = 1.0            # seconds per example
CLIP_SAMPLES = int(SAMPLE_RATE * DURATION_S)
BATCH_SIZE = 64
EPOCHS = 10
SEED = 1337

# -----------------------
# Discover data
# -----------------------
commands = [c for c in tf.io.gfile.listdir(DATA_DIR) if tf.io.gfile.isdir(DATA_DIR/c)]
commands = sorted(commands)
if TARGET not in commands:
    raise ValueError(f"Expected a folder '{TARGET}' inside {DATA_DIR}. Found: {commands}")

# Build files list
files = tf.io.gfile.glob(str(DATA_DIR/'*'/'*.wav'))
random.Random(SEED).shuffle(files)

# Binary labels: 1 for 'secret', 0 for everything else
def file_to_label(path):
    label = 1 if pathlib.Path(path).parent.name == TARGET else 0
    return label

labels = [file_to_label(f) for f in files]

# Stratified-ish split
def split_idxs(y, train=0.8, val=0.1):
    idxs = list(range(len(y)))
    # keep positive/negative ratio in each split
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]
    random.Random(SEED).shuffle(pos); random.Random(SEED).shuffle(neg)
    def take(parts, frac):
        n = int(len(parts)*frac); return parts[:n], parts[n:]
    pos_tr, pos_rem = take(pos, train)
    neg_tr, neg_rem = take(neg, train)
    pos_v,  pos_te = take(pos_rem, val/(1-train) if 1-train>0 else 1.0)
    neg_v,  neg_te = take(neg_rem, val/(1-train) if 1-train>0 else 1.0)
    train_idx = pos_tr + neg_tr
    val_idx   = pos_v  + neg_v
    test_idx  = pos_te + neg_te
    random.Random(SEED).shuffle(train_idx)
    random.Random(SEED).shuffle(val_idx)
    random.Random(SEED).shuffle(test_idx)
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = split_idxs(labels)
def select(lst, idxs): return [lst[i] for i in idxs]
train_files, val_files, test_files = select(files, train_idx), select(files, val_idx), select(files, test_idx)

# -----------------------
# Audio -> Log-Mel Spectrogram
# -----------------------
# Helper: decode, pad/trim to 1s, resample if needed (assumes most clips are 16 kHz)
@tf.function
def load_wav_1s(path):
    audio_bin = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(audio_bin, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    # If sample rate is not known to be 16k, you could resample (omitted for speed).
    # Pad/trim to fixed length
    n = tf.shape(audio)[0]
    audio = tf.cond(n < CLIP_SAMPLES,
                    lambda: tf.pad(audio, [[0, CLIP_SAMPLES - n]]),
                    lambda: audio[:CLIP_SAMPLES])
    return audio

# STFT -> Mel -> log
# Using parameters similar to TF tutorials
NUM_MEL_BINS = 40
FRAME_LENGTH = 640    # 40 ms @ 16k
FRAME_STEP   = 320    # 20 ms @ 16k
FFT_LENGTH   = 1024

mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MEL_BINS,
    num_spectrogram_bins=FFT_LENGTH//2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=80.0,
    upper_edge_hertz=7600.0
)

@tf.function
def wav_to_logmelspec(waveform):
    stft = tf.signal.stft(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, fft_length=FFT_LENGTH)
    spectrogram = tf.abs(stft)**2
    mel_spec = tf.matmul(spectrogram, mel_filterbank)
    # log( mel + epsilon )
    log_mel = tf.math.log(mel_spec + 1e-6)
    # Normalize per-example
    mean = tf.math.reduce_mean(log_mel)
    std  = tf.math.reduce_std(log_mel) + 1e-9
    log_mel = (log_mel - mean) / std
    # Add channel dim
    return tf.expand_dims(log_mel, -1)

def path_to_example(path):
    y = 1 if tf.strings.split(path, os.sep)[-2] == TARGET else 0
    wav = load_wav_1s(path)
    x = wav_to_logmelspec(wav)
    return x, tf.cast(y, tf.int32)

def make_ds(paths, shuffle=False, cache=True):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle: ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(path_to_example, num_parallel_calls=tf.data.AUTOTUNE)
    if cache: ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(train_files, shuffle=True)
val_ds   = make_ds(val_files)
test_ds  = make_ds(test_files)

# -----------------------
# Model: tiny CNN
# -----------------------
inputs = tf.keras.Input(shape=(None, NUM_MEL_BINS, 1))  # time x mel x ch
x = tf.keras.layers.Conv2D(16, (5,5), padding="same", activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D((2,2))(x)
x = tf.keras.layers.Conv2D(48, (3,3), padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
)
model.summary()

# -----------------------
# Train
# -----------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight={0:1.0, 1: max(1.0, len([l for l in labels if l==0]) / max(1, len([l for l in labels if l==1])))},  # handle imbalance
    verbose=2
)

# -----------------------
# Evaluate
# -----------------------
print("\nTest set evaluation:")
model.evaluate(test_ds, verbose=2)

# -----------------------
# Inference helper
# -----------------------
def score_file(wav_path):
    x, _ = path_to_example(wav_path)
    x = tf.expand_dims(x, 0)  # batch=1
    p = float(model.predict(x, verbose=0)[0][0])
    return p

# Example usage:
# prob = score_file("mini_speech_commands/secret/your_sample.wav")
# print(f\"P(secret) = {prob:.3f}; detected? {prob >= 0.5}\")
