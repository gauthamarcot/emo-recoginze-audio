import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def extract_features(file_path):
    # Extracting features using librosa
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    return mfccs

def load_data(dataset_path):
    # Loading data and extracting features
    X, y = [], []
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = subdir + os.path.sep + file
            if file_path.endswith(".mp3"):
                emotion = subdir.split("/")[-1]
                features = extract_features(file_path)
                X.append(features)
                y.append(emotion)
    return X, y
    
    def train_model(X, y):
    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initializing the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Training the model
    model.fit(X_train, y_train)
    # Evaluating the model
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)
    return model

def predict_emotion(model, file_path):
    features = extract_features(file_path)
    features = np.array(features).reshape(1,-1)
    prediction = model.predict(features)
    return prediction[0]

def main():
    dataset_path = "data/voices"
    X, y = load_data(dataset_path)
    model = train_model(X, y)
    file_path = "path/to/audio.mp3"
    emotion = predict_emotion(model, file_path)
    print("Predicted emotion: ", emotion)

if __name__ == "__main__":
    main()
