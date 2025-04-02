from abc import ABC, abstractmethod

class Problem(ABC):
    @abstractmethod
    def generate_solution(self, num_samples=1):
        pass

    @abstractmethod
    def fitness(self, solution):
        pass