from abc import ABC, abstractmethod

class Problem(ABC):
    @abstractmethod
    def generate_solution(self, num_samples=1):
        pass

    @abstractmethod
    def fitness(self, solutions):
        pass

    def fitness_omp(self, solutions):
        raise NotImplementedError("OpenMP fitness not implemented")

    def fitness_gpu(self, solutions):
        raise NotImplementedError("GPU fitness not implemented")
    
    def fitness_hybrid(self, solutions, speedup=1):
        raise NotImplementedError("Hybrid fitness not implemented")

def patch_problem(problem):
    #Comprobar si el argumento forma parte de la clase Problem o de la clase hija
    if not isinstance(problem, Problem):
        raise TypeError("The argument must be an instance of the Problem class or its subclasses.")
    def add_method(func):
        if not callable(func):
            raise TypeError("The argument must be a callable function.")
        setattr(problem, func.__name__, func)
        return func
        