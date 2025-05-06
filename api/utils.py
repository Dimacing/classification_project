ratings = {
    "simple_nn": [],
    "random_forest": [],
    "logistic_regression": [],
    "transformer": []
}

def add_rating(model_name: str, rating: int):
    if model_name in ratings:
        if 1 <= rating <= 5:
            ratings[model_name].append(rating)
        else:
            print(f"Warning: Invalid rating {rating} passed to add_rating for {model_name}.")
    else:
         print(f"Warning: Attempted to add rating for unknown model key: '{model_name}'. Known keys: {list(ratings.keys())}")


def get_average_ratings():
    averages = {}
    for model_name in ratings.keys():
        rates = ratings.get(model_name, [])
        if rates:
            averages[model_name] = round(sum(rates) / len(rates), 2)
        else:
            averages[model_name] = 0.0
    return averages