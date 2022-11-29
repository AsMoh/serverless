import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import os


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


    
    
MODEL_NAME=os.getenv('MODEL_NAME','dino-vs-dragon-v2.tflite')

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

print(f" input_index is : {input_index}")
print(f" output_index is : {output_index}")


data={ 'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'
    }


target_size= 150


classes = [
    'dino',
    'dragon'
]

def predict(url):
    img=download_image(url)
    img=prepare_image(img, (target_size,target_size))
    x=np.array(img,dtype='float32')
    X = np.array([x])
    X=X/255.0
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result


