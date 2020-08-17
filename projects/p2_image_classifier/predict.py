import argparse
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

image_size = 224


parser = argparse.ArgumentParser(description='Flower image classifier command line application.')
parser.add_argument('image_path', metavar='image_path', type=str,
                    help='path of the image to be identified')
parser.add_argument('save_model', metavar='save_model', type=str,
                    help='model that has to be used')
parser.add_argument('--top_k', metavar='top_k', type=int,default=1,
                    help='Return the top K most likely classes:')
parser.add_argument('--category_names', metavar='category_names', type=str,default=None,
                    help='Path to a JSON file mapping labels to flower names:')
args = parser.parse_args()

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict(path_image, model, top):
    img = Image.open(path_image)
    test_image = np.asarray(img)
    processed_test_image = process_image(test_image)
    batched_test_image = np.expand_dims(processed_test_image, axis=0)
    predictions = model.predict(batched_test_image)[0]
    probs, classes = tf.nn.top_k (predictions, top)
    return probs.numpy(), classes.numpy()+1


load_model = tf.keras.models.load_model(args.save_model, custom_objects={'KerasLayer':hub.KerasLayer})
probs, classes = predict(args.image_path,load_model,args.top_k)

class_names = dict()
if(args.category_names):
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
print("======================")
for i in range(len(classes)):
    label=class_names.get(str(classes[i]),classes[i])
    print(label,probs[i])
print("======================")
