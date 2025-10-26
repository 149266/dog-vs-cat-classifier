import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- Data augmentering for trening ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# --- Treningsgeneratorer ---
train_group1_gen = train_datagen.flow_from_directory(
    "dog_vs_cat_data_split2/train_group1.1",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

train_group2_gen = train_datagen.flow_from_directory(
    "dog_vs_cat_data_split2/train_group2.2",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# --- Valideringsgenerator ---
val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dog_vs_cat_data_split2/validation.2",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --- Funksjon for balansert batch ---
def balanced_batch_generator(gen1, gen2):
    while True:
        x1, y1 = next(gen1)
        x2, y2 = next(gen2)
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        # Shuffle batch slik at katt/hund er blandet
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        yield x[idx], y[idx]

train_generator = balanced_batch_generator(train_group1_gen, train_group2_gen)

# --- Lag CNN-modellen ---
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Early stopping ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# --- Beregn steps per epoch ---
steps_per_epoch = (train_group1_gen.samples + train_group2_gen.samples) // 32
validation_steps = val_generator.samples // 32

# --- Tren modellen ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=20,
    callbacks=[early_stop]
)

# --- Plot trenings- og valideringsnøyaktighet ---
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# --- Evaluer på valideringsdata ---
eval_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dog_vs_cat_data_split2/validation.2",
    target_size=(64,64),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

y_pred_probs = model.predict(eval_generator)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = eval_generator.classes

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))
