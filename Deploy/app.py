import os
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
import cv2
from camviz import grad_cam
from models import get_model

app = Flask(__name__)

STATIC_FOLDER = '/home/rajat/Documents/Aegis/CAPSTONE/Data - CXR8/Data/Deployment/Final/static'
# STATIC_FOLDER = 'D:\Aegis\Capstone\App\static'

# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'

# Path to the folder where we store the different models
MODEL_FOLDER = STATIC_FOLDER + '/models'

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = grad_cam(fullname, model, graph)

        heatmap_filename = file.filename+'heatmap.png'
        heatmap = os.path.join(UPLOAD_FOLDER, heatmap_filename)

        # cv2.imwrite(UPLOAD_FOLDER + 'heatmap.png', result[heatmap])
        if os.path.exists(heatmap):
            os.remove(heatmap)
            cv2.imwrite(heatmap, result['heatmap'])
        else:
            cv2.imwrite(heatmap, result['heatmap'])

        return render_template('predict.html', image_file_name=heatmap_filename, 
                                label = result['prediction'], 
                                accuracy = result['accuracy'])


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':

    file_path = MODEL_FOLDER + '/best_weights_dice_0.4061.hdf5'
    
    model = get_model()
    model.load_weights(file_path)

    graph = tf.get_default_graph()

    app.run(debug=True)