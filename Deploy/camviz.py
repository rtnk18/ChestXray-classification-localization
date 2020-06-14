import numpy as np
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from keras import backend as K
import cv2

def grad_cam(path, model, graph):
    ''' 
        The function will generate a prediction and localization heatmap
        for the input image.
    ''' 

    # List of diseases
    cats_list = ['Atelectasis','Cardiomegaly','Effusion','Infiltration',
 				 'Mass','No Finding','Nodule','Pneumonia','Pneumothorax']

    # Loading and preprocessing image as per DenseNet format
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    with graph.as_default():
        # Making prediction and getting model output for predicted class
        preds = model.predict(x)
        preds = list(preds.flatten())
        class_idx = np.argmax(preds)
        class_output = model.output[:, class_idx]

        # Getting final convolution layer and calculating gradients
        last_conv_layer = model.get_layer('relu')
        grads = K.gradients(class_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], 
                            [pooled_grads, 
                            last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        for i in range(1024):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
            
        # Creating heatmap from calculated gradients
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # Superimposing heatmap on original image
        img = cv2.imread(path)
        img = cv2.resize(img,(512, 512))
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return {'prediction':cats_list[class_idx], 
            'heatmap':superimposed_img, 
            'accuracy':np.round(preds[class_idx]*100,2)}