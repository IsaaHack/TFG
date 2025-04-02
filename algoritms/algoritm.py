from abc import ABC, abstractmethod

class Algoritm(ABC):
    @abstractmethod
    def check_required_methods(self):
        pass

    @abstractmethod
    def fit(self):
        pass