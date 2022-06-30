from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import base64
import cv2

model = load_model('my_model.h5')
class_label=["Actinic keratoses/ Bowen's Disease ",
            "Basal Cell Carcinoma(Cancerous)", 
            "Seborrheic Keratosis", 
            "Dermatofibrom B", 
            "Melanoma(Cancerous)", 
            "Melanocytic Nevi", 
            "Vascular Lesions"]

class_desc=["Actinic keratoses are pre-malignant but transformation to in-situ or invasive squamous cell carcinoma (SCC) is rare. Specific treatment is not essential, particularly in mild disease.",
            "Basal cell carcinoma is a type os skin cancer often appears as a slightly transparent bump on the skin, though it can take other forms.",
            "Seborrheic keratosis (seb-o-REE-ik ker-uh-TOE-sis) is a common noncancerous (benign) skin growth. People tend to get more of them as they get older.",
            "Dermatofibroma is a commonly occurring cutaneous entity usually centered within the skin's dermis. Dermatofibromas are referred to as benign fibrous histiocytomas of the skin.",
            "Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin. Knowing the warning signs of skin cancer can help ensure that cancerous changes are detected and treated before the cancer has spread. Melanoma can be treated successfully if it is detected early.",
            "Melanocytic Nevi is a skin condition characterized by an abnormally dark, noncancerous skin patch (nevus) that is composed of pigment-producing cells called melanocytes.",
            "Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks. There are three major categories of vascular lesions: Hemangiomas, Vascular Malformations, and Pyogenic Granulomas."]

app=Flask(__name__)

def predict_cancer(image):


    img = Image.open(image)     #reading the image
    img = img.resize((224,224))     #resizing the image
    img = np.array(img)     #converting the image to numpy array
    #img = np.expand_dims(img, axis=0)   #expanding the dimension
    #img = img[:,:,:,0:3]

    # input_img = Image.open(image)
    # input_img = np.array(input_img)  

    #input_img = cv2.imread(image)
    #input_img1 = cv2.resize(input_img,(224,224))
    input_img1 = np.reshape(img,[1,224,224,3])
    input_img1 = tf.keras.applications.mobilenet.preprocess_input(input_img1)
    classes_pred = model.predict(input_img1)
    classes = [np.argmax(element) for element in classes_pred]
    prediction= classes
    confidence = max(classes_pred)[np.argmax(classes_pred)]*100
    return class_label[prediction[0]],class_desc[prediction[0]], confidence.round(2)

@app.route('/',methods=['POST','GET'])
@app.route('/home', methods=['POST','GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST','GET'])
def result():
    img = request.files['img'] 
    #parr = np.frombuffer(img, np.uint8)
    
    prediction, description, confidence = predict_cancer(img)
    img = Image.open(img) 
    data = io.BytesIO()
    img.save(data, "JPEG")
    encoded_img = base64.b64encode(data.getvalue())
    decoded_img = encoded_img.decode('utf-8')
    img_data = f"data:image/jpeg;base64,{decoded_img}"
    return render_template('result.html', data=[img_data, prediction, confidence, description])

if __name__=='__main__':
    app.run(debug=True)