import cv2
import numpy as np
import tensorflow as tf
from models import yolo_v3_model

classes_path = './coco.names'
weights_path = './checkpoints/yolov3.tf'
img_size = 416
image_path = './input_city.jpg'
output_path = './output.jpg'
num_classes = 80


def preprocess_image(x_train, size):
    x_train = tf.expand_dims(x_train, 0)
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
    return img

yolo = yolo_v3_model()

yolo.load_weights(weights_path).expect_partial()

class_names = [c.strip() for c in open(classes_path).readlines()]

img_raw = tf.image.decode_image(
    open(image_path, 'rb').read(), channels=3)

img = preprocess_image(img_raw, img_size)

boxes, scores, classes, nums = yolo(img)

for i in range(nums[0]):
    print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))

img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
cv2.imwrite(output_path, img)
print('output saved to: {}'.format(output_path))

