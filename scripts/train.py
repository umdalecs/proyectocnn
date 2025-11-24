from pathlib import Path
import os 
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

root_dir = Path(__file__).resolve().parent.parent

def main():
  TRAIN_DATA_DIR = os.path.join(root_dir, "dataset", "train")
  VAL_DATA_DIR = os.path.join(root_dir, "dataset", "val")
  MODEL_PATH = os.path.join(root_dir, "model","face_classifier.h5")

  IMG_SIZE = (224, 224)
  BATCH = 32
  NUM_CLASSES = 5  # ajuste según dataset

  # Generators con augmentation ligero en entrenamiento
  train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=15,
      width_shift_range=0.05,
      height_shift_range=0.05,
      zoom_range=0.1,
      horizontal_flip=True,
      brightness_range=(0.8,1.2),
  )

  val_datagen = ImageDataGenerator(rescale=1./255)

  train_gen = train_datagen.flow_from_directory(TRAIN_DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')
  val_gen = val_datagen.flow_from_directory(VAL_DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

  # Base model
  base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights='imagenet')
  base.trainable = False

  x = base.output
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.4)(x)
  x = layers.Dense(256, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

  model = models.Model(inputs=base.input, outputs=out)
  model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
  rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

  # Entrenar head
  model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[es, rlr])

  # Fine-tune: descongelar algunas capas
  base.trainable = True
  for layer in base.layers[:-60]:
      layer.trainable = False

  model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[es, rlr])

  # Guardar
  model.save(MODEL_PATH)

if __name__ == "__main__":
  main()