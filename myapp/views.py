from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image


def home(request):
    return render(request, "index.html")


def about(request):
    return render(request, "about.html")


model_path = os.path.join(os.getcwd(), "models/1")
model = load_model(model_path)

class_name = ['freshapples', 'freshbanana', 'freshoranges',
              'rottenapples', 'rottenbanana', 'rottenoranges']


def fruit_prediction(request):
    image_filename = None
    if request.method == 'POST':
        f = request.FILES['fruit']
        filename = f.name
        s = "fruitimages/"
        des = os.path.join('myapp/static/', s, filename)  # Correct file path
        with open(des, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        image_filename = os.path.join(
            'myapp/static/', s, filename)  # to access image
        imgfilename = "static/fruitimages/"+filename
        # print("::::::file name is : :::::", imgfilename)

        test_image = image.load_img(image_filename, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = model.predict(test_image)

        predicted_class = class_name[np.argmax(prediction[0])]
        confidence = round(np.max(prediction[0])*100)

        return render(request, "fruit_prediction.html", {"confidence": "Prediction -> "+str(confidence) + "%",
                                                         "prediction": "Fruit name -> "+str(predicted_class),
                                                         "image_filename": imgfilename})
    else:
        return render(request, "fruit_prediction.html")


model_path = os.path.join(os.getcwd(), "veg_models/1")
veg_models = load_model(model_path)
veg_class_name = [
    'Fresh Capsicum',
    'Fresh Carrot',
    'Fresh Cucumber',
    'Fresh Potato',
    'Fresh Tomato',
    'Rotten Capsicum',
    'Rotten Carrot',
    'Rotten Cucumber',
    'Rotten Potato',
    'Rotten Tomato'
]
# vegetable


def veg_prediction(request):
    image_filename = None
    if request.method == 'POST':
        f = request.FILES['veg']
        filename = f.name
        s = "vegImages/"
        des = os.path.join('myapp/static/', s, filename)  # Correct file path
        with open(des, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        image_filename = os.path.join('myapp/static/', s, filename)
        imgfilename = "static/vegImages/"+filename
        print("::::::file name is : :::::", imgfilename)

        test_image = image.load_img(image_filename, target_size=(300, 300))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        prediction = veg_models.predict(test_image)
        predicted_class = veg_class_name[np.argmax(prediction[0])]
        confidence = round(np.max(prediction[0])*100)

        return render(request, "veg_prediction.html", {"confidence": "Prediction -> "+str(confidence) + "%",
                                                       "prediction": "Vegetable Name -> "+str(predicted_class), "image_filename": imgfilename})
    else:
        return render(request, "veg_prediction.html")
