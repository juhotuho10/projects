# Sound classification project using a CNN model
# The project is setup in a way that it loads up the saved model when run with initial conditions

from keras import layers, models
import os
import zipfile
import io
from pydub import AudioSegment
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from scipy.stats import norm
import keras
import random
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt

# generate data and quite or load the processed data from disk and train with it, or load model from disk
# "data", "train" or "load"
RUNMODE = "load"
N_DATA_SHARDS = 5


def load_zip(zip_path: str, skip_words: list[str]) -> list[AudioSegment]:
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zip_file_audio = []
        for member in zipf.namelist():
            for word in skip_words:
                if word in member:
                    print(f"skipping: {member}")
                    continue
            _, ext = os.path.splitext(member)
            ext = ext.lower()
            if ext not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"} or ext == ".txt":
                continue

            try:
                data_bytes = zipf.read(member)
            except KeyError:
                # skip weird zip entries
                continue

            bio = io.BytesIO(data_bytes)
            fmt = ext.lstrip(".")
            try:
                audio_seg: AudioSegment = AudioSegment.from_file(bio, format=fmt)
            except Exception:
                bio.seek(0)
                try:
                    audio_seg = AudioSegment.from_file(bio)
                except Exception:
                    print(f"skip {member}")
                    continue

            zip_file_audio.append(audio_seg)
            # print(audio_seg.dBFS)
        return zip_file_audio


def load_sounds(folder_path, skip_words) -> list[AudioSegment]:
    # load all zip files in folder into a list of audiosegments
    container = []
    for file_name in os.listdir(folder_path):
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext != ".zip":
            continue
        zpath = os.path.join(folder_path, file_name)
        audio_files = load_zip(zpath, skip_words)
        container.extend(audio_files)
    return container


def process_sounds(
    car_sound_data: list[AudioSegment], tram_sound_data: list[AudioSegment], is_training_data: bool
) -> tuple[np.ndarray, np.ndarray]:
    # for processing sounds into X and y data
    # includes the whole processing pipeline except data normalization, which is done later

    def np_audio(audio_seg: AudioSegment) -> tuple[np.ndarray, int]:  # type: ignore
        # convert to AudioSegment np.ndarray float32 in [-1,1]
        samples = np.array(audio_seg.get_array_of_samples())
        if audio_seg.channels > 1:
            samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)
        # convert to float32 in [-1,1]
        max_val = float(1 << (8 * audio_seg.sample_width - 1))
        samples = samples.astype(np.float32) / max_val
        return samples, audio_seg.frame_rate

    def zscore_cutoffs(scores: list, low_pct: float, high_pct: float):
        mean = np.mean(scores)
        std = np.std(scores)

        low = mean + norm.ppf(low_pct) * std
        high = mean + norm.ppf(high_pct) * std

        return low, high

    def initial_filtering(sounds: list[AudioSegment]) -> list[AudioSegment]:
        # some samples have width of 8 which is unsuitable for us
        sounds = [s for s in sounds if s.sample_width in (1, 2, 3, 4)]

        loudness = [s.dBFS for s in sounds]

        # loud and quiet sounds out
        # sounds that aren't good length out
        # must have sampling rate > 44000

        # filter out 10% of lowest and 5% highest values
        # more from low side since it has more noise
        low, high = zscore_cutoffs(loudness, 0.10, 0.95)
        sounds = [s for s in sounds if ((low < s.dBFS < high) and (4.7 < s.duration_seconds < 7) and (s.frame_rate > 44000))]

        return sounds

    # filter out training data but not test data
    if is_training_data:
        car_sound_data = initial_filtering(car_sound_data)
        tram_sound_data = initial_filtering(tram_sound_data)

    @dataclass
    class AudioData:
        data: np.ndarray
        sr: int
        is_car: bool

    combined_sounds: list[AudioData] = []

    combined_sounds.extend([AudioData(*np_audio(s), True) for s in car_sound_data])
    combined_sounds.extend([AudioData(*np_audio(s), False) for s in tram_sound_data])

    def lowpass_filter(sound: AudioData) -> AudioData:
        nyquist = 0.5 * sound.sr
        cutoff = 20_000.0
        cutoff = cutoff / nyquist

        b, a = butter(4, cutoff, btype="low")  # type: ignore
        sound.data = filtfilt(b, a, sound.data)
        return sound

    combined_sounds = [lowpass_filter(s) for s in combined_sounds]

    def resample(audio: AudioData) -> AudioData:
        # we know that all audio is > 44000hz sampling rate from filtering
        target_sr = 44100  # samples from 44100 and 48000 sr, we want all to be 44100 sr
        if audio.sr == target_sr:
            return audio
        audio.data = librosa.resample(audio.data, orig_sr=audio.sr, target_sr=target_sr)
        audio.sr = target_sr
        return audio

    combined_sounds = [resample(s) for s in combined_sounds]

    def to_log_mel(audio: AudioData) -> AudioData:
        # use log mel spectogram for image in CNN training
        n_mels = 128
        n_fft = 1024
        n_frames = 550
        seq_len = int(len(audio.data) / n_frames)

        mel = librosa.feature.melspectrogram(
            y=audio.data.astype(np.float32),
            sr=audio.sr,
            n_fft=n_fft,
            hop_length=seq_len,
            window="hann",
            n_mels=n_mels,
            fmin=0.0,
            fmax=None,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

        audio.data = log_mel
        return audio

    combined_sounds = [to_log_mel(s) for s in combined_sounds]

    # cutting off the start and the top of the graph, it seems to be anomalous in some data
    # and limit all to be 100 x 500
    combined_sounds = [AudioData(s.data[:100, 50:550], s.sr, s.is_car) for s in combined_sounds]
    assert all(s.data.shape == (100, 500) for s in combined_sounds)  # all same shape

    def pipeline_data(sounds: list[AudioData]) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([s.data for s in sounds])
        y = np.array([s.is_car for s in sounds], dtype=np.int32)

        return (X, y)

    random.shuffle(combined_sounds)

    train_X, train_y = pipeline_data([s for s in combined_sounds])

    if is_training_data:
        X_rev = train_X[:, :, ::-1]
        # add reversed data to the set to make more data
        # the audio being in reverse shouldn't be problematic, but doubles our training data
        train_X = np.concatenate([train_X, X_rev], axis=0)
        train_y = np.concatenate([train_y, train_y], axis=0)

    return (train_X, train_y)


def generate_normalization(train_X: np.ndarray) -> tuple[np.float32, np.float32]:
    return (np.mean(train_X).astype(np.float32), np.std(train_X).astype(np.float32))


def normalize_data(X: np.ndarray, train_mean: np.float32, train_std: np.float32) -> np.ndarray:
    return (X - train_mean) / (train_std)


def predict_accuracy(model: keras.Model, X: np.ndarray, y: np.ndarray, detailed: bool) -> np.ndarray:
    # use model to predict data and print out the accuracy and predction statistics
    pred = model.predict(X, verbose=0)  # type: ignore

    pred = (pred > 0.5).astype(int).flatten()

    acc = np.mean(pred == y)
    print(f"accuracy: {acc:.2f}")
    print(f"prediction mean: {np.mean(pred):.2f}, prediction std: {np.std(pred):.2f}")

    if detailed:
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))

        # precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"precision: {precision:.2f}")
        print(f"recall: {recall:.2f}")

    print()
    return pred


if RUNMODE == "data":
    print("process data, save it to the disk then exit")

    car_path = "others_sounds\\car"
    tram_path = "others_sounds\\tram"

    # my car sounds and validation that is not in training set
    my_car_path = "my_sounds\\car"
    my_tram_path = "my_sounds\\tram"

    test_car_path = "test_sounds\\car"
    test_tram_path = "test_sounds\\tram"

    banned_terms = ["-bus", "_bus", "-truck", "_truck"]
    # no tram sounds in cars, and car sounds in tram, they are sometimes mixed up
    train_car_sounds: list[AudioSegment] = load_sounds(car_path, ["_tram", "-tram", "-ratikka", "_ratikka"] + banned_terms)
    train_tram_sounds: list[AudioSegment] = load_sounds(tram_path, ["_car", "_auto", "-car", "-auto"] + banned_terms)

    # don't filter anything in test and validation
    my_car_sounds: list[AudioSegment] = load_sounds(my_car_path, [])
    my_tram_sounds: list[AudioSegment] = load_sounds(my_tram_path, [])

    test_car_sounds: list[AudioSegment] = load_sounds(test_car_path, [])
    test_tram_sounds: list[AudioSegment] = load_sounds(test_tram_path, [])

    train_X, train_y = process_sounds(train_car_sounds, train_tram_sounds, is_training_data=True)

    test_X, test_y = process_sounds(my_car_sounds, my_tram_sounds, is_training_data=False)

    val_X, val_y = process_sounds(test_car_sounds, test_tram_sounds, is_training_data=False)

    train_mean, train_std = generate_normalization(train_X)

    train_X = normalize_data(train_X, train_mean, train_std)
    test_X = normalize_data(test_X, train_mean, train_std)
    val_X = normalize_data(val_X, train_mean, train_std)

    with open("train_mean_std.pkl", "wb") as f:
        pickle.dump((train_mean, train_std), f)

    # save the training data in N = N_DATA_SHARDS different groups
    # this is because using all of the data at once took too much ram, so we just individually load it in packets
    train_dir = "./data_dir/train_data"

    os.makedirs(train_dir, exist_ok=True)
    rng = np.random.default_rng(123)
    n = train_X.shape[0]
    idx = rng.permutation(n)
    X_shuf = train_X[idx]
    y_shuf = train_y[idx]
    sizes = [(n // N_DATA_SHARDS) + (1 if i < (n % N_DATA_SHARDS) else 0) for i in range(N_DATA_SHARDS)]
    start = 0
    for i, s in enumerate(sizes):
        end = start + s
        shard_path = os.path.join(train_dir, f"data_{i}.npz")
        np.savez_compressed(shard_path, X=X_shuf[start:end], y=y_shuf[start:end])
        start = end

    np.savez_compressed("./data_dir/test_data.npz", X=test_X, y=test_y)
    np.savez_compressed("./data_dir/val_data.npz", X=val_X, y=val_y)
    print("done")
    exit()
elif RUNMODE == "train":
    print("train model")
    d = np.load("./data_dir/test_data.npz")
    test_X, test_y = d["X"], d["y"]

    d = np.load("./data_dir/val_data.npz")
    val_X, val_y = d["X"], d["y"]

    lr = 5e-4
    batch_size = 32
    epochs = 30
    base_layer_size = 32

    def build_cnn(input_shape=(100, 500, 1)):
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Conv2D(base_layer_size, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(base_layer_size, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(2 * base_layer_size, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.Conv2D(2 * base_layer_size, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(4 * base_layer_size, (3, 3), padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Dropout(0.3),
                layers.GlobalAveragePooling2D(),
                layers.Dense(2 * base_layer_size, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model

    model = build_cnn()
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),  # type: ignore
        loss="binary_crossentropy",
        metrics=["accuracy", "mse"],
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True, save_weights_only=False, verbose=1
    )

    def get_balanced_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # balances data to be 50 / 50

        y = np.ravel(y)

        idx_0 = np.where(y == 0.0)[0]
        idx_1 = np.where(y == 1.0)[0]
        assert len(idx_0) != 0 and len(idx_1) != 0
        n = min(len(idx_0), len(idx_1))
        sampled_0 = np.random.choice(idx_0, n, replace=False)
        sampled_1 = np.random.choice(idx_1, n, replace=False)

        balanced_idx = np.concatenate([sampled_0, sampled_1])
        np.random.shuffle(balanced_idx)

        assert sum(y[balanced_idx] == 0.0) == sum(y[balanced_idx] == 1.0)

        return X[balanced_idx], y[balanced_idx]

    callbacks = [checkpoint]

    history_dict = {"loss": [], "accuracy": [], "mse": [], "val_loss": [], "val_accuracy": [], "val_mse": []}

    for epoch in range(epochs):
        # training only on parts of data because the whole dataset was too much ram to load in at once
        set_i = epoch % N_DATA_SHARDS
        print(f"epoch: {epoch}, data batch num: {set_i}")
        d = np.load(f"./data_dir/train_data/data_{set_i}.npz")
        loaded_x, loaded_y = d["X"], d["y"]

        train_x, train_y = get_balanced_data(loaded_x, loaded_y)

        history = model.fit(
            train_x, train_y, epochs=1, batch_size=batch_size, callbacks=callbacks, shuffle=True, validation_data=(val_X, val_y)
        )

        # gather training statistics
        history_dict["loss"].append(history.history["loss"][0])
        history_dict["accuracy"].append(history.history["accuracy"][0])
        history_dict["mse"].append(history.history["mse"][0])
        history_dict["val_loss"].append(history.history["val_loss"][0])
        history_dict["val_accuracy"].append(history.history["val_accuracy"][0])
        history_dict["val_mse"].append(history.history["val_mse"][0])

        print("test accuracy")
        predict_accuracy(model, test_X, test_y, False)

    print("validation accuracy")
    predict_accuracy(model, test_X, test_y, True)

    print("test accuracy")
    predict_accuracy(model, val_X, val_y, True)

    # plotting training statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history_dict["loss"], label="Training Loss")
    axes[0].plot(history_dict["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Model Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history_dict["accuracy"], label="Training Accuracy")
    axes[1].plot(history_dict["val_accuracy"], label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Model Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history_dict["mse"], label="Training MSE")
    axes[2].plot(history_dict["val_mse"], label="Validation MSE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE")
    axes[2].set_title("Model MSE")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
    plt.show()

elif RUNMODE == "load":
    # use the model to predict data
    # load in .zip files, extract the audio from inside of them, run them through the pipeline, normalize them and then predict them

    model: keras.Model = keras.models.load_model("my_best_model.keras")  # type: ignore

    my_car_path = "./cars-driving-past-in-rain.zip"
    my_tram_path = "./tram-arriving-leaving-station.zip"

    # load
    my_car_sounds: list[AudioSegment] = load_zip(my_car_path, [])
    my_tram_sounds: list[AudioSegment] = load_zip(my_tram_path, [])

    # process
    test_X, test_y = process_sounds(my_car_sounds, my_tram_sounds, is_training_data=False)

    # normalize
    with open("train_mean_std.pkl", "rb") as f:
        train_mean, train_std = pickle.load(f)
        test_X = normalize_data(test_X, train_mean, train_std)

    print(test_X.shape, test_y.shape)

    print("validation accuracy")
    predicted = predict_accuracy(model, test_X, test_y, True)
    print("1 = car, 0 = tram")
    print(f"predicted values: {predicted}")

    # also an example of loading in the npz data and predicting on them
    # the .npz data should already be normalized so we dont need to do anything to it
    for i in range(N_DATA_SHARDS):
        try:
            d = np.load(f"./data_dir/train_data/data_{i}.npz")
        except Exception as _:
            exit("could not load training data, we dont have training data generated")
        loaded_x, loaded_y = d["X"], d["y"]
        print(loaded_x.shape, loaded_y.shape)
        print(f"train accuracy in batch {i}")
        predict_accuracy(model, loaded_x, loaded_y, True)


else:
    print('invalid runmode, use: "data", "train" or "load"')
