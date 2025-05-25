# recommender.py

import random

def recommend_hair(face_shape):
    """
    Recommends a hairstyle overlay file based on the given face shape.
    """
    hair_options = {
        "round": ["assets/round_hair_1.png", "assets/round_hair_2.png"],
        "oval": ["assets/oval_hair_1.png", "assets/oval_hair_2.png"],
        "square": ["assets/square_hair_1.png", "assets/square_hair_2.png"],
        "heart": ["assets/heart_hair_1.png", "assets/heart_hair_2.png"],
        "diamond": ["assets/diamond_hair_1.png", "assets/diamond_hair_2.png"],
        "triangle": ["assets/triangle_hair_1.png", "assets/triangle_hair_2.png"],
        "oblong": ["assets/oblong_hair_1.png", "assets/oblong_hair_2.png"]
    }
    options = hair_options.get(face_shape.lower(), [])
    if not options:
        return None
    return random.choice(options)
