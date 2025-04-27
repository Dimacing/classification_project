ratings = {
    "simple_nn": [],
    "distilbert": []
}

def add_rating(model_name: str, rating: int):
    if model_name in ratings:
        ratings[model_name].append(rating)


def get_average_ratings():
    averages = {}
    for model_name, rates in ratings.items():
        if rates:
            averages[model_name] = sum(rates) / len(rates)
        else:
            averages[model_name] = 0.0
    return averages

