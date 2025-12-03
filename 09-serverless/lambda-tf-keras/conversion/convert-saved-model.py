from tensorflow import keras

model = keras.models.load_model('clothing_classification.keras')
model.export("clothing_classification_saved_model")
