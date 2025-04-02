from abc import ABC, abstractmethod

class Algoritm(ABC):
    @abstractmethod
    def fit(self):
        pass