import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, scrolledtext

import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS

# Paths for DL model
train_dir = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\hackTHON\Soil types'
validation_dir = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\hackTHON\Soil types'

# Define the extract_features function
def extract_features(generator, model):
    print("Extracting features...")
    features = model.predict(generator, verbose=1)
    features = features.reshape((features.shape[0], -1))  # Flatten the features
    labels = generator.classes
    print("Features extracted.")
    return features, labels

# DL: Load the EfficientNetB4 model for feature extraction
print("Loading EfficientNetB4 model...")
efficientnet_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("EfficientNetB4 model loaded.")

# Adding dropout and pooling layers to the base model
x = efficientnet_base.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
model = tf.keras.models.Model(inputs=efficientnet_base.input, outputs=x)

# Function to classify a new image
def classify_soil(image_path, model, feature_extractor):
    print(f"Classifying image: {image_path}")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    features = feature_extractor.predict(img_array)
    features = features.flatten().reshape(1, -1)
    prediction = model.predict(features)
    class_labels = list(train_generator.class_indices.keys())
    print(f"Classified as: {class_labels[prediction[0]]}")
    return class_labels[prediction[0]]

# Function to load and preprocess images using ImageDataGenerator
def create_data_generators(train_dir, validation_dir):
    print("Creating data generators...")
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
    )
    validation_generator = datagen.flow_from_directory(
        validation_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
    )
    print("Data generators created.")
    return train_generator, validation_generator

# DL: Create data generators
train_generator, validation_generator = create_data_generators(train_dir, validation_dir)

# DL: Extract features and train the XGBoost classifier on soil types
train_features, train_labels = extract_features(train_generator, model)
print("Training XGBoost classifier...")
xgb_classifier = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_classifier.fit(train_features, train_labels)
print("XGBoost classifier trained.")

# ML: Convert CSV to pickle if not exists
csv_file_path = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\crop\files\agriculture_data.csv'
pkl_file_path = r'C:\Users\BHUVANA S\AppData\Local\Programs\Python\Python312\crop\files\agriculture_data.pkl'

if not os.path.exists(pkl_file_path):
    if os.path.exists(csv_file_path):
        print("Converting CSV to pickle...")
        data = pd.read_csv(csv_file_path)
        data.to_pickle(pkl_file_path)
        print("CSV converted to pickle.")

# ML: Load the DataFrame from the pickle file
if os.path.exists(pkl_file_path):
    print("Loading data from pickle file...")
    data = pd.read_pickle(pkl_file_path)
    print("Data loaded.")

# ML: Preprocess data and encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for column in ['Crop', 'Season', 'State', 'Fertilizer', 'Pesticide', 'Soil_Type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
print("Categorical variables encoded.")

# ML: Train-test split
X = data[['Soil_Type']]
y_crop = data['Crop']
y_yield = data['Yield']
y_fertilizer = data['Fertilizer']
y_season = data['Season']

print("Splitting data...")
X_train, X_test, y_crop_train, y_crop_test, y_yield_train, y_yield_test, y_fertilizer_train, y_fertilizer_test, y_season_train, y_season_test = train_test_split(
    X, y_crop, y_yield, y_fertilizer, y_season, test_size=0.2, random_state=42)
print("Data split.")

# ML: Train RandomForest models for crop, yield, fertilizer, and season predictions
print("Training RandomForest models...")
rf_crop = RandomForestClassifier(n_estimators=100, random_state=42)
rf_yield = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fertilizer = RandomForestClassifier(n_estimators=100, random_state=42)
rf_season = RandomForestClassifier(n_estimators=100, random_state=42)

rf_crop.fit(X_train, y_crop_train)
rf_yield.fit(X_train, y_yield_train)
rf_fertilizer.fit(X_train, y_fertilizer_train)
rf_season.fit(X_train, y_season_train)
print("RandomForest models trained.")

# Function to get additional information based on predicted soil type
def get_additional_info(soil_type_encoded):
    print(f"Getting additional information for soil type: {soil_type_encoded[0, 0]}")
    filtered_data = data[data['Soil_Type'] == soil_type_encoded[0, 0]]
    
    # Extract relevant columns
    soil_types = label_encoders['Soil_Type'].inverse_transform(filtered_data['Soil_Type']).tolist()
    states = label_encoders['State'].inverse_transform(filtered_data['State']).tolist()
    seasons = label_encoders['Season'].inverse_transform(filtered_data['Season']).tolist()
    yields = [f"{yield_value:.2f}" for yield_value in filtered_data['Yield']]
    fertilizers = label_encoders['Fertilizer'].inverse_transform(filtered_data['Fertilizer']).tolist()
    crops = label_encoders['Crop'].inverse_transform(filtered_data['Crop']).tolist()

    # Determine the maximum length of columns
    max_length = max(len(soil_types), len(states), len(seasons), len(yields), len(fertilizers), len(crops))
    
    # Extend lists to match the maximum length
    soil_types.extend([None] * (max_length - len(soil_types)))
    states.extend([None] * (max_length - len(states)))
    seasons.extend([None] * (max_length - len(seasons)))
    yields.extend([None] * (max_length - len(yields)))
    fertilizers.extend([None] * (max_length - len(fertilizers)))
    crops.extend([None] * (max_length - len(crops)))

    # Create DataFrame with the specified order of columns
    info_df = pd.DataFrame({
        'Soil_Type': soil_types,
        'State': states,
        'Season': seasons,
        'Yield': yields,
        'Fertilizer': fertilizers,
        'Crops': crops
    })
    
    return info_df.to_string(index=False)  # Convert DataFrame to string

# Function to make predictions based on image (DL -> ML)
def make_predictions(image_path):
    print("Making predictions...")
    # DL: Classify soil type from image
    soil_type = classify_soil(image_path, xgb_classifier, model)
    print("Soil type:", soil_type)
    
    # Convert soil_type to its encoded value for ML model input
    soil_type_encoded = label_encoders['Soil_Type'].transform([soil_type])
    soil_type_encoded = np.array(soil_type_encoded).reshape(-1, 1)
    
    # ML: Predict crop, yield, fertilizer, and season based on the soil type
    crop = label_encoders['Crop'].inverse_transform(rf_crop.predict(soil_type_encoded))[0]
    yield_amount = rf_yield.predict(soil_type_encoded)[0]
    fertilizer = label_encoders['Fertilizer'].inverse_transform(rf_fertilizer.predict(soil_type_encoded))[0]
    season = label_encoders['Season'].inverse_transform(rf_season.predict(soil_type_encoded))[0]
    
    # Get additional information
    additional_info = get_additional_info(soil_type_encoded)
    
    print("Predictions made.")
    return crop, yield_amount, fertilizer, season, additional_info

# Function to select an image file and make predictions
def image():
    image_path = filedialog.askopenfilename(title='Select an image', filetypes=[('Image files', '*.jpg *.jpeg *.png')])
    if image_path:
        crop, yield_amount, fertilizer, season, additional_info = make_predictions(image_path)
        display_results(crop, yield_amount, fertilizer, season, additional_info)

# Function to display the results in a Tkinter scrolled text widget
def display_results(crop, yield_amount, fertilizer, season, additional_info):
    results_window = tk.Tk()
    results_window.title("Prediction Results")
    
    results_text = scrolledtext.ScrolledText(results_window, wrap=tk.WORD, width=80, height=20)
    results_text.pack(expand=True, fill='both')
    
    results = f"Predicted Crop: {crop}\n"
    results += f"Predicted Yield: {yield_amount:.2f}\n"
    results += f"Recommended Fertilizer: {fertilizer}\n"
    results += f"Recommended Season: {season}\n\n"
    
    results += "Additional Information:\n"
    results += additional_info
    
    results_text.insert(tk.END, results)
    results_text.config(state=tk.DISABLED)
    
    results_window.mainloop()

# Main function to run the Tkinter app
#def main():
#    root = tk.Tk()
#    root.title("Soil Image Classification")
    
#    select_button = tk.Button(root, text="Select Soil Image", command=select_image)
#    select_button.pack(pady=20)
    
#    root.mainloop()


app = Flask(__name__)
CORS(app)  # To handle CORS issues during development

@app.route('/output', methods=['POST'])
def process_file():
    # Get the file from the form data
    image = request.files.get('image')

    # Simulate some processing
    if image:
        print(f"Received image: {image.filename}")
        image_path = os.path.join('uploaded_images', image.filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save the image to the specified path
        image.save(image_path)
        print(f"Image saved at {image_path}")

        crop, yield_amount, fertilizer, season, additional_info = make_predictions(image_path)

    else:
        print("No image received")

    # Return a response to the client
    return jsonify({
        "status": "success",
        "message": "File processed successfully",
        "crop": crop,
        "yield": yield_amount,
        "fertilizer": fertilizer,
        "season": season,
        "additional_info": additional_info
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)