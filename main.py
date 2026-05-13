import re
import math
import numpy as np

def lemmatize(word, physics_terms, chemistry_terms):
    if word in physics_terms:
        return physics_terms[word]
    elif word in chemistry_terms:
        return chemistry_terms[word]
    else:
        return word

def preprocess(text1):
    lowercase = text1.lower()
    remove_punctuation = re.sub(r'[^\w\s]', '', lowercase)
    token1 = remove_punctuation.split()
    stop_words = ["it", "is", "a", "the", "and", "or", "but", "if", "in", "on", "with", "as", "by", "for", "to", "of", "its", "that", "this", "these", "those", "he", "she", "they", "we", "you", "me", "him", "her", "them", "my", "your", "his", "her", "their", "our", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"]
    physics_terms = {"forces":"force", "accelerations":"acceleration", "velocities": "velocity", "speeds": "speed", "momenta": "momentum", "energies": "energy", "powers": "power", "masses": "mass", "weights": "weight", "distances": "distance", "displacements": "displacement", "times": "time", "frictions": "friction", "collisions": "collision", "waves": "wave", "frequencies": "frequency", "wavelengths": "wavelength", "amplitudes": "amplitude", "echoes": "echo", "currents": "current", "voltages": "voltage", "charges": "charge", "circuits": "circuit", "resistors": "resistor", "capacitors": "capacitor", "cells": "cell", "batteries": "battery", "switches": "switch", "conductors": "conductor", "insulators": "insulator", "moving": "move", "moves": "move", "moved": "move", "accelerating": "accelerate", "accelerated": "accelerate", "accelerates": "accelerate", "colliding": "collide", "collided": "collide", "collides": "collide", "measuring": "measure", "measured": "measure", "measures": "measure", "calculating": "calculate", "calculated": "calculate", "calculates": "calculate", "conducting": "conduct", "conducted": "conduct", "conducts": "conduct", "charging": "charge", "charged": "charge", "faster": "fast", "fastest": "fast", "slower": "slow", "slowest": "slow", "heavier": "heavy", "heaviest": "heavy", "lighter": "light", "lightest": "light", "positive": "positive", "negative": "negative", "conductive": "conductive", "resistive": "resistive"}
    chemistry_terms = {"atoms": "atom", "molecules": "molecule", "elements": "element", "compounds": "compound", "mixtures": "mixture", "hydrocarbons": "hydrocarbon", "alkanes": "alkane", "alkenes": "alkene", "alcohols": "alcohol", "polymers": "polymer", "reactions": "reaction", "reactants": "reactant", "products": "product", "equilibriums": "equilibrium", "temperatures": "temperature", "pressures": "pressure", "concentrations": "concentration", "reacting": "react", "reacted": "react", "reacts": "react", "mixing": "mix", "mixed": "mix", "mixes": "mix", "heating": "heat", "heated": "heat", "heats": "heat", "cooling": "cool", "cooled": "cool", "cools": "cool", "acidic": "acidic", "basic": "basic", "stable": "stable", "unstable": "unstable", "organic": "organic", "inorganic": "inorganic"}

    removed_stop_words = [word for word in token1 if word not in stop_words]
    lemmatized_words = [lemmatize(word, physics_terms, chemistry_terms) for word in removed_stop_words]
    return " ".join(lemmatized_words)

def n_grams(text, n):
    tokens = text.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class TfidVectorizer:
    def __init__(self):
        pass

    def calculate_tf(self, term, document):
        return document.count(term) / len(document)

    def calculate_idf(self, term, documents):
        doc_frequency = sum(1 for doc in documents if term in doc)
        return math.log(1 + len(documents) / (1 + doc_frequency))

    def fit(self, documents):
        self.tfidf_model = {}
        self.tfidf_model['terms'] = list(set(term for doc in documents for term in doc))
        self.tfidf_model['idf'] = {}
        for term in self.tfidf_model['terms']:
            self.tfidf_model['idf'][term] = self.calculate_idf(term, documents)

    def transform(self, doc):
        tfidf_vector = []
        for term in self.tfidf_model['terms']:
            tf = self.calculate_tf(term, doc)
            idf = self.tfidf_model['idf'][term]
            tfidf_vector.append(tf * idf)
        return np.asarray(tfidf_vector)

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return None
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.sqrt(np.sum(vec1 ** 2))
    magnitude_vec2 = np.sqrt(np.sum(vec2 ** 2))
    return dot_product / (magnitude_vec1 * magnitude_vec2)

