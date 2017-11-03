# Make an intermediate output model to check filters
from keras import Model
def makeIntermediate(keras_model,layer_name):
    # keras_model: keras model that has been trained previously
    # layer_name: name of the layer with the desired output

    # Define the new model that stops at the desired layer
    intermediate_layer_model = Model(inputs=keras_model.input,\
                                     outputs=keras_model.get_layer(layer_name).output)

    # Return the newly defined model
    return intermediate_layer_model