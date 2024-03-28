def save_model(model, category):
    model.save(f"model_{category.lower()}.keras")
