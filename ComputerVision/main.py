# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import scipy.io
import argparse
import time

import SmartVision as SV
import color_classification as cc
from SmartVision.utils import load_model

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Fine tunes a rsnet 152 classification ' + \
                                    'model on Standford dataset')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-i", "--image",action="store_true", help="path to input image")
    parser.add_argument("-p", "--path", default="data/test/", help="test images location")
    parser.add_argument("-f", "--filename", default="00001.jpg", help="test image filename")
    group.add_argument("-r", "--randomimage",action="store_true", help="path to input random test images")
    parser.add_argument("-s", "--size",type=int, default=10, help="random sample size")
    args = parser.parse_args()
    
    img_width, img_height = 224, 224
    model = load_model()
    model.load_weights('models/model.96-0.89.hdf5')
    print("[INFO] loading Fine tuned Rsnet152 model from disk...")

    cars_meta = scipy.io.loadmat('devkit/cars_meta')
    class_names = cars_meta['class_names']  # shape=(1, 196)
    class_names = np.transpose(class_names)
    
    start = time.time()
    if args.randomimage:
        test_path = 'data/test/'
        test_images = [f for f in os.listdir(test_path) if 
                       os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]     
        num_samples = args.size
        samples = random.sample(test_images, num_samples)
    else:
        samples = [args.filename]
        test_path = args.path
        
    results = []    
    for i, image_name in enumerate(samples):
        filename = os.path.join(test_path, image_name)
        print('Start processing image: {}'.format(filename))
        bgr_img = cv.imread(filename)
        bgr_img = cv.resize(bgr_img, (img_width, img_height), cv.INTER_CUBIC)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = np.expand_dims(rgb_img, 0)
        preds = model.predict(rgb_img)
        prob_make = np.max(preds)
        #print(prob_make)
        class_id = np.argmax(preds)
        text = ('Predict: {}, prob: {}'.format(class_names[class_id][0][0], prob_make))
        color_class =cc.car_color_classification.color_classifier(filename)
        #print(color_class)
        results.append({'Vehical Class': class_names[class_id][0][0], 'prob_make': float(prob_make), 
                      'color class': color_class[0]['color'], 'prob_color': float(color_class[0]['prob'])})
        cv.imwrite('images/{}_out.png'.format(i), bgr_img)
    print(results)
    end = time.time()
    print("[INFO] make classifier took {:.6f} seconds".format(end - start))
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

    K.clear_session()
