# train_cnn.py
import os, glob, pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array, load_img

DATASET_DIR = "dataset"
IMG_SIZE = (160,160)  # small input for speed
BATCH = 16
EPOCHS = 12
MODEL_PATH = "models/model.h5"
LE_PATH = "models/label_encoder.pkl"

def load_images_labels():
    images = []
    labels = []
    for person in os.listdir(DATASET_DIR):
        pdir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(pdir): continue
        for img_path in glob.glob(os.path.join(pdir, "*.jpg")):
            img = load_img(img_path, target_size=IMG_SIZE)
            x = img_to_array(img)
            images.append(x)
            labels.append(person)
    images = np.array(images, dtype="float32") / 255.0
    return images, labels

def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    # freeze base for initial training
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    X, y = load_images_labels()
    if len(X)==0:
        raise SystemExit("No images found in dataset/ — run prepare_dataset.py first.")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    with open(LE_PATH, "wb") as f:
        pickle.dump(le, f)
    num_classes = len(le.classes_)
    print("Classes:", le.classes_)

    # basic augmentation
    aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.1, zoom_range=0.1, horizontal_flip=True)

    model = build_model(num_classes)
    # callbacks
    chk = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1)
    # train/val split
    from sklearn.model_selection import train_test_split
    Xtrain, Xval, ytrain, yval = train_test_split(X, y_enc, test_size=0.15, stratify=y_enc, random_state=42)

    model.fit(aug.flow(Xtrain, ytrain, batch_size=BATCH),
              validation_data=(Xval, yval),
              epochs=EPOCHS,
              steps_per_epoch=max(1, len(Xtrain)//BATCH),
              callbacks=[chk, early])
    print("Saved model and label encoder.")
