from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

#loads in the model
model_path = os.path.join('Model-Files', 'model-v4.h5')
model = load_model(model_path)

#function for preprocessing the image
def preprocessImage(imagePath, targetSize=(224, 224)):
    img = load_img(imagePath, target_size=targetSize)
    imgArray = img_to_array(img) / 255.0
    imgArray = np.expand_dims(imgArray, axis=0)
    return imgArray

#route to the main website page
@app.route('/')
def home():
    return render_template('Website.html')

#route to the prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file'] #takes in file from request
    
    file_path = os.path.join('static', file.filename)
    file.save(file_path)    #file path to loacte uploaded image

    processedImage = preprocessImage(file_path)
    output = model.predict(processedImage)  #predicts inputted image
    
    classLabels = ['NON-RECYCLABLE', 'ORGANIC', 'RECYCLABLE']
    imageType = classLabels[np.argmax(output)]  #image type set to a label
    
    return jsonify({'class': imageType})    #returns the read the label from imageType

if __name__ == '__main__':
    app.run(debug=True)