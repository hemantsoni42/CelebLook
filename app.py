from flask import Flask, render_template, request
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as pI_V3
import pickle
import tensorflow as tf
import cv2
import os
from mtcnn import MTCNN
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading


app = Flask(__name__)

detector = MTCNN()
model_V3 = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
model_50 = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list_V3, filenames_V3, feature_list_50, filenames_50 = [
    pickle.load(open(f, 'rb')) for f in [
        'embedding_V3.pkl', 'merged_filenames_V3.pkl',
        'embedding.pkl', 'merged_filenames_50.pkl'
    ]
]

def main_code(features, model):
    predictions = []
    if model == 'V3':
        indices = recommend(feature_list_V3, features)
        predictions = [" ".join(filenames_V3[index].split('\\')[1].split('_')) for index in indices]
    elif model == '50':
        indices = recommend(feature_list_50, features)
        predictions = [" ".join(filenames_50[index].split('\\')[1].split('_')) for index in indices]
    return predictions

def extract_features(img_path, model, detector, model_value):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    min_size = min(img.shape[:2])
    factor = 0.707
    results = []
    for scale in [1.0, factor, factor ** 2]:
        new_size = int(scale * min_size)
        img_resized = cv2.resize(img, (new_size, new_size))
        results_resized = detector.detect_faces(img_resized)
        for result in results_resized:
            result['box'] = [int(coord / scale) for coord in result['box']]
            results.append(result)

    if not results:
        return None

    result = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, width, height = result['box']
    x -= int(width * 0.1)
    y -= int(height * 0.1)
    width = int(width * 1.2)
    height = int(height * 1.2)
    img_height, img_width, _ = img.shape
    x = max(x, 0)
    y = max(y, 0)
    width = min(width, img_width - x)
    height = min(height, img_height - y)
    face = img[y:y + height, x:x + width]
    face = cv2.resize(face, (224, 224))
    face = tf.keras.preprocessing.image.img_to_array(face)
    face = np.expand_dims(face, axis=0)
    if model_value == 'V3':
        face = pI_V3(face)
    else:
        face = preprocess_input(face)
    result = model.predict(face).flatten()
    return result

def recommend(feature_list, features):
    similarity_cos = cosine_similarity(features.reshape(1, -1), feature_list)[0]
    similarity_euclidean = np.linalg.norm(features - feature_list, axis=1)
    similarity_manhattan = np.sum(np.abs(features - feature_list), axis=1)
    similarity_combined = (similarity_cos - np.min(similarity_cos)) / (
            np.max(similarity_cos) - np.min(similarity_cos)) - \
                          (similarity_euclidean / np.max(similarity_euclidean)) - \
                          (similarity_manhattan / np.max(similarity_manhattan))
    indices = np.argsort(similarity_combined)[::-1]
    unique_indices = []
    for i in indices:
        if i not in unique_indices:
            unique_indices.append(i)
        if len(unique_indices) == 3:
            break
    return unique_indices

@app.route('/', methods=['GET', 'POST'])
def celebrity_predictor():
    if request.method == 'POST':
        uploaded_image = request.files['image']
        if uploaded_image:
            file_path = os.path.join('uploads', uploaded_image.filename)
            uploaded_image.save(file_path)
            # display_image = Image.open(file_path)
            features_V3 = None
            features_50 = None

            def process_model_V3():
                nonlocal features_V3
                features_V3 = extract_features(os.path.join('uploads', uploaded_image.filename), model_V3, detector, 'V3')

            def process_model_50():
                nonlocal features_50
                features_50 = extract_features(os.path.join('uploads', uploaded_image.filename), model_50, detector, '50')

            # Create threads to process the models concurrently
            thread_V3 = threading.Thread(target=process_model_V3)
            thread_50 = threading.Thread(target=process_model_50)

            # Start the threads
            thread_V3.start()
            thread_50.start()

            # Wait for both threads to finish
            thread_V3.join()
            thread_50.join()
            if features_50 is None and features_V3 is None:
                os.remove(file_path)
                return render_template('result.html', result="No face is detected")
            else:
                features_V3_predictions = []                
                features_50_predictions = []
                def features_50_predictions_function():
                    nonlocal features_50_predictions
                    features_50_predictions = main_code(features_50, '50')
                def features_V3_predictions_function():
                    nonlocal features_V3_predictions
                    features_V3_predictions = main_code(features_V3, 'V3')
                thread_features_50 = threading.Thread(target=features_50_predictions_function)
                thread_features_V3 = threading.Thread(target=features_V3_predictions_function)

                thread_features_V3.start()
                thread_features_50.start()

                thread_features_V3.join()
                thread_features_50.join()

                all_predictions = features_50_predictions + features_V3_predictions
                sorted_predictions = sorted(set(all_predictions), key=all_predictions.count, reverse=True)
    
                images = []
                images_name = []
                for i in sorted_predictions:
                    image_path = 'static/data/' + i + '.jpg'
                    images_name.append(i)
                    images.append(image_path)

                image_data = list(zip(images, images_name))
                os.remove(file_path)
                return render_template('result.html', image_data=image_data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)