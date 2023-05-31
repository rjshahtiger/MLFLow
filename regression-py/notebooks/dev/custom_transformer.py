from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['population_per_household'] = X['population'] / X['households']
        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
            return X

        else:
            return X
