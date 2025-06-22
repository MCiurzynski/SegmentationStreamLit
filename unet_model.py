import tensorflow as tf
import os

train_images_path = 'data/ISIC2018_Task1-2_Training_Input'
train_masks_path = 'data/ISIC2018_Task1_Training_GroundTruth'
validate_images_path = 'data/ISIC2018_Task1-2_Validation_Input'
validate_masks_path = 'data/ISIC2018_Task1_Validation_GroundTruth'
test_images_path = 'data/ISIC2018_Task1-2_Test_Input'
test_masks_path = 'data/ISIC2018_Task1_Test_GroundTruth'

def get_unet_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    def upsample(filters, size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]


    inputs = tf.keras.layers.Input(shape=[224, 224, 3])

    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

class Augment(tf.keras.layers.Layer):
  def __init__(self, seed=42):
    super().__init__()
    self.flip_input = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed)
    self.flip_label = tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed)
    self.rotate_input = tf.keras.layers.RandomRotation(0.1, seed=seed+1)
    self.rotate_label = tf.keras.layers.RandomRotation(0.1, seed=seed+1)
    self.random_brightness_input = tf.keras.layers.RandomBrightness(0.2, seed=seed+2)
    self.random_brightness_label = tf.keras.layers.RandomBrightness(0.2, seed=seed+2)
    self.random_contrast_input = tf.keras.layers.RandomContrast(0.2, seed=seed+3)
    self.random_contrast_label = tf.keras.layers.RandomContrast(0.2, seed=seed+3)

  def call(self, inputs, labels):
    inputs = self.flip_input(inputs)
    labels = self.flip_label(labels)
    inputs = self.rotate_input(inputs)
    labels = self.rotate_label(labels)
    inputs = self.random_brightness_input(inputs)
    labels = self.random_brightness_label(labels)
    inputs = self.random_contrast_input(inputs)
    labels = self.random_contrast_label(labels)
    return inputs, labels

def get_process_path(image_size):
  def process_path(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [image_size, image_size])
    mask = tf.cast(mask, tf.float32) / 255.0 

    return image, mask
  return process_path

def get_dataset(path_to_images, path_to_masks, batch_size=32, is_train=True, image_size=224):
  image_files = sorted([os.path.join(path_to_images, fname) for fname in os.listdir(path_to_images) if fname.lower().endswith(('.png', '.jpg'))])
  mask_files = sorted([os.path.join(path_to_masks, fname) for fname in os.listdir(path_to_masks) if fname.lower().endswith(('.png', '.jpg'))])

  dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
  dataset = dataset.map(get_process_path(image_size), num_parallel_calls=tf.data.AUTOTUNE).cache()

  if is_train:
    dataset = dataset.shuffle(buffer_size=len(image_files), seed=42)

  dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return dataset

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'Augment': Augment})

if __name__ == "__main__":
    model = get_unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    train_dataset = get_dataset(train_images_path, train_masks_path, batch_size=32, is_train=True, image_size=224)
    val_dataset = get_dataset(validate_images_path, validate_masks_path, batch_size=32, is_train=False, image_size=224)
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)
    model.save('unet_model.h5')