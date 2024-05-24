import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_sample(images, boxes, pboxes=None, plot_predictions=False):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap='binary')
        center = (boxes[i, 0]*75, boxes[i, 1]*75)
        plt.gca().add_patch(Rectangle(center,
                                      (boxes[i, 2]-boxes[i, 0])*75,
                                      (boxes[i, 3]-boxes[i, 1])*75,
                                      edgecolor='red',
                                      facecolor='none',
                                      lw=4))
        if plot_predictions:
            center = (pboxes[i, 0]*75, pboxes[i, 1]*75)
            plt.gca().add_patch(Rectangle(center,
                                        (pboxes[i, 2]-pboxes[i, 0])*75,
                                        (pboxes[i, 3]-pboxes[i, 1])*75,
                                        edgecolor='blue',
                                        facecolor='none',
                                        lw=4))
    plt.show()


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['classifier_loss'], label='classifier loss')
    ax1.plot(history.history['val_classifier_loss'], label='val classifier loss')
    ax2.plot(history.history['bounding_box_loss'], label='bounding box loss')
    ax2.plot(history.history['val_bounding_box_loss'], label='val bounding box loss')
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.show()


def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x


def classifier(inputs):
    return tf.keras.layers.Dense(10, activation='softmax', name='classifier')(inputs)


def regressor(inputs):
    return tf.keras.layers.Dense(4, name='bounding_box')(inputs)


def create_model():
    inputs = tf.keras.layers.Input(shape=(75, 75, 1,))
    dense_output = feature_extractor(inputs)
    classification_output = classifier(dense_output)
    regression_output = regressor(dense_output)
    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, regression_output])
    return model


training_images = np.load('../training_images.npy')
training_boxes = np.load('../training_boxes.npy')
training_labels = np.load('../training_labels.npy')

validation_images = np.load('../validation_images.npy')
validation_boxes = np.load('../validation_boxes.npy')
validation_labels = np.load('../validation_labels.npy')

plot_sample(training_images, training_boxes)

model = create_model()
model.compile(optimizer='adam',
              loss={
                  'classifier': 'categorical_crossentropy',
                  'bounding_box': 'mse'
              },
              metrics={
                  'classifier': 'acc',
                  'bounding_box': 'mse'
              })
model.summary()
history = model.fit(training_images, (training_labels, training_boxes),
                    validation_data=(validation_images, (validation_labels, validation_boxes)),
                    epochs=10)

plot_history(history)
predictions = model.predict(validation_images)
plot_sample(validation_images, validation_boxes, predictions[1], plot_predictions=True)


def check_iou(iou, iou_threshold=0.6):
    good = 0
    bad = 0
    for i in iou:
        if i >= iou_threshold:
            good +=1
            continue
        bad += 1
    return good, bad


def intersection_over_union(true_box, pred_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred =  np.split(pred_box, 4, axis = 1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis = 1)

    smoothing_factor = 1e-12

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum((xmax_overlap - xmin_overlap), 0)  * np.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area

    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou


iou = intersection_over_union(validation_boxes, predictions[1])
good, bad = check_iou(iou)
print('Number of good bounding box prediction: ', good)
print('Number of bad bounding box prediction: ', bad)
