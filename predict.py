import tensorflow as tf
import os, sys
import numpy as np

def _predict(p_this, loaded_model):
    # This predicts some random values I plucked out of the Q O and V's csvs
    prediction = loaded_model.predict(p_this)
    preds = []
    for result in prediction:
        index = np.argmax(result)

        #print(index)
        preds.append(chr(index+65))

    return preds

if __name__ == "__main__":
    # Provide the path to the directory where the model is saved
    model_path = os.curdir + "/asl_model.keras"

    # Load the saved model
    loaded_model = tf.keras.models.load_model(model_path)

    p_this = np.array(sys.argv[1]).reshape(1, 63)
    #p_this.reshape(1, 63)

    print(_predict(p_this, loaded_model))