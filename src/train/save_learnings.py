import pickle


def save_learnings(model, category, labels):
    model.save(f"model_{category.lower()}.keras")

    with open(f"labels_{category.lower()}.pickle", "wb") as f:
        pickle.dump(labels, f)
