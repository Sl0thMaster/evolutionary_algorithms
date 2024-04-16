import tensorflow as tf
import matplotlib.pyplot as plt
# from PIL import Image, ImageEnhance
import numpy as np


data_dir = 'part_of_dataset/'
# batch_size = 12
img_height = 144
img_width = 256
validation_split = 0.1

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # color_mode='grayscale',
  validation_split=validation_split,
  subset="training",
  seed=123,
  image_size=(img_height, img_width))
  # batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  # color_mode='grayscale',
  validation_split=validation_split,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width))
  # batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)



plt.figure(figsize=(16, 9))
for images, labels in train_ds.take(1):
  for i in range(12):
    ax = plt.subplot(4, 4, i + 1)
    img = images[i].numpy().astype(np.uint8)
    plt.imshow(img) #, cmap=plt.cm.gray)
    plt.title(class_names[labels[i]])
    plt.axis("off")


# # CNN Model

# In[3]:


def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

train_ds = train_ds.map(normalize)
val_ds = val_ds.map(normalize)

train_ds = train_ds.cache()
val_ds = val_ds.cache()


# In[4]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (17, 17), activation='relu', input_shape=(144, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (9, 9), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# model.summary()


# In[ ]:


model.fit(train_ds, epochs=10, batch_size=1)


# In[ ]:


model.save_weights('./model_weights/weights_0')


# In[ ]:


val_loss, val_acc = model.evaluate(val_ds)
print('Accuracy on test dataset:', val_acc)

