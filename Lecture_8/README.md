# DL laboratory 8

### Prof. Stefano Carrazza

**Summary:** Data augmentation and transfer learning.

## Exercise 1: Classification with data augmentation

1. Load the following dataset with 3670 photos of flowers with shape (180, 180, 3) each.
    ```python
    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    ```

2. Construct a training and validation dataset using `tf.keras.utils.image_dataset_from_directory`, 20% split, 32 batch sizes. Inspect the training dataset by plotting image samples.

3. Build a CNN classifier by applying a rescaling layer (normalizing by 255), 3 convolutional layers with [15, 32, 64] filters, 3x3 kernels, ReLU activations, interchanged with max pooling layers. After flattening, apply a dense layer with 128 nodes and ReLU activation.

4. Perform a fit for few epochs (< 10), monitor and plot the evolution of the loss function and accuracy for the train and validation set. In order to optimize dataset performance cache and prefetch the original datasets with:
    ```python
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    ```

5. Build a sequential model with data augmentation layers, in particular with horizontal random flip, random rotation (0.1) and random zoom (0.1). Plot samples of this layer. Attach this layer at the beginning of the previous model and introduce a dropout layer before flatten in order to reduce overfitting.

## Exercise 2: Transfer learning

1. Load the following dataset containing thousands of images of cats and dogs:
    ```python
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = pathlib.Path(path) / 'train'
    validation_dir = pathlib.Path(path) / 'validation'
    ```

2. Construct a training and validation dataset using `tf.keras.utils.image_dataset_from_directory`, 20% split, 32 batch sizes. Inspect the training dataset by plotting image samples. Prefetch data following the same approach implemented in the previous exercise.

3. Construct a data augmentation model with horizontal random flip and rotation (0.2). Plot samples of augmented data.

4. Allocate a MobileNetV2 base model passing the input image shape, excluding the classification layers at the top of the model (`include_top=False`) and using weights from ImageNet (`weights='imagenet'`). Freeze this model by calling `base_model.trainable = False`.

5. Construct the final model using the functional API. The input passes through: a data augmentation layer, a preprocessing input (which normalizes images for MobileNetV2) using `tf.keras.applications.mobilenet_v2.preprocess_input`, the freeze base model MobileNetV2, an average over the spatial locations using `tf.keras.layers.GlobalAveragePooling2D()`, a dropout layer (0.2), and finally a dense layer with a single unit. The network output should be considered as logit, i.e. positive numbers predict class 1 while negative class 0.

6. Train the model with Adam and learning rate 1e-4, binary cross-entropy loss function and few epochs (< 10). Monitor and plot the loss function and accuracy for each epoch, for training and validation sets.
