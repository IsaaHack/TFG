from abc import ABC, abstractmethod

class Problem(ABC):
    @abstractmethod
    def fitness(self, solution):
        pass