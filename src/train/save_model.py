from keras import saving

def save_model(model):
    saving.save_model(model, save_format="h5")