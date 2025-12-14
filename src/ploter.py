import visualkeras
from tensorflow import keras
# Assuming 'model' is your compiled Keras model
model = keras.models.load_model('models/lstm_forecaster.h5')
visualkeras.layered_view(model, to_file='model_layered_view.png')