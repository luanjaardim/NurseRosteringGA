from parse import parse_txt
import random
import numpy as np
random.seed(42)

class NurseRosteringGA:
    def __init__(self,
        problem_instance,
        pop_size=50,
        generations=200,
        crossover_rate=0.7,
        mutation_rate=0.05,
        elitism=1,
        penalities_weights=[100] * 10
    ):
        assert len(penalities_weights) == 10
        self.staff_num = len(problem_instance['staff'])
        self.days_num = problem_instance['len_day']
        # Number of covers to choose a staff to
        self.individual_size = len(problem_instance['cover'])
        self.instance_data = problem_instance
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.penalities_weights = penalities_weights

        # Initializing maps to convert between id and a index
        self.shift_to_index = dict()
        self.index_to_shift = dict()
        self.employee_to_index = dict()
        self.index_to_employee = dict()
        for i, shift in enumerate(problem_instance['shifts']):
            self.shift_to_index[shift['id']] = i
            self.index_to_shift[i] = shift['id']
        for i, emp in enumerate(problem_instance['staff']):
            self.employee_to_index[emp['id']] = i
            self.index_to_employee[i] = emp['id']


    # ============================================================
    #   1. Problem-Specific Functions (YOU MODIFY THESE)
    # ============================================================
    def get_shift_from_id(self, id):
        index = self.shift_to_index[id]
        return self.instance_data['shifts'][index]

    def get_employee_from_id(self, id):
        index = self.employee_to_index[id]
        return self.instance_data['staff'][index]

    def generate_individual(self):
        """ Create a random candidate solution. """
        return np.random.randint(low=0, high=self.staff_num, size=(self.individual_size,))


    def objective_function(self, indiv):
        """
        Compute the objective value.
        Lower is better (minimization).
        Modify this function with your real objective.
        """

    def pen_employee_working_more_per_day(self, indiv):
        """
        An employee working more than one shift per day, first constraint
        """
        counter = np.zeros((self.staff_num, self.days_num,))
        for i, cover in enumerate(self.instance_data['cover']):
            counter[indiv[i]][cover['day']] += 1
        return np.sum(counter > 1) # Count the days that an employee works twice or more

    def pen_shift_rotation(self, indiv):
        mat_shift_types = []
        for _ in range(self.staff_num): mat_shift_types.append([set() for _ in range(self.days_num)])
        for i, cover in enumerate(self.instance_data['cover']):
            # each element of this matrix is a set of the shifts of an employee at some day
            mat_shift_types[indiv[i]][cover['day']].add(cover['id'])

        penalities = 0
        for i, cover in enumerate(self.instance_data['cover'][1:], start=1):
            day = cover['day']
            shift = self.get_shift_from_id(cover['id'])
            if shift['cannot_follow']:
                for s in shift['cannot_follow']:
                    # verify if the worker worked this same shift 's' on the previous day
                    if s in mat_shift_types[indiv[i]][day-1]:
                        penalities += 1
        return penalities 


    def pen_maximum_shift_types(self, indiv):
        counter = np.zeros((self.staff_num, len(self.instance_data['shifts']),))
        for i, cover in enumerate(self.instance_data['cover']):
            # counts the number of each shift type for each worker
            counter[indiv[i]][self.shift_to_index[cover['id']]] += 1
        print(counter)
        penalities = 0
        for emp in self.instance_data['staff']:
            for (s_id, limit_) in emp['shift_limits']:
                limit = int(limit_)
                emp_index = self.employee_to_index[emp['id']]
                shi_index = self.shift_to_index[s_id]

                # verify if the limit was reached
                if counter[emp_index][shi_index] > limit:
                    penalities += counter[emp_index][shi_index] - limit
        return penalities

    def pen_min_max_working_time(self, indiv):
        pass


    def constraint_penalty(individual):
        """
        Compute penalties for violating constraints.
        Return 0 if no violations.
        Increase value for worse constraint violations.
        """
        pass


    def fitness(individual):
        """
        Final fitness = objective + penalties.
        """
        pass
        # return objective_function(individual) + constraint_penalty(individual)


    # ============================================================
    #   2. Genetic Operators (Selection, Crossover, Mutation)
    # ============================================================

    def tournament_selection(population, k=3):
        """Pick best of k random individuals."""
        pass
        # candidates = random.sample(population, k)
        # return min(candidates, key=fitness)


    def crossover(parent1, parent2, rate=0.7):
        """Uniform crossover."""
        pass
        # if random.random() > rate:
        #     return parent1[:], parent2[:]
        #
        # child1, child2 = [], []
        # for a, b in zip(parent1, parent2):
        #     if random.random() < 0.5:
        #         child1.append(a); child2.append(b)
        #     else:
        #         child1.append(b); child2.append(a)
        #
        # return child1, child2


    def mutate(individual, rate=0.05):
        """Random reset mutation."""
        # for i in range(len(individual)):
        #     if random.random() < rate:
        #         individual[i] = random.randint(0, 10)    # modify with your domain
        # return individual
        pass


    # ============================================================
    #   3. Main GA Loop
    # ============================================================

    def run(self):
        # ---- create initial population ----
        population = [self.generate_individual() for _ in range(self.pop_size)]
        print(len(population), population[0].shape)
        # best = min(population, key=fitness)
        #
        # for gen in range(generations):
        #
        #     new_population = []
        #
        #     # ---- elitism: keep best individuals ----
        #     sorted_pop = sorted(population, key=fitness)
        #     new_population.extend(sorted_pop[:elitism])
        #
        #     # ---- create new individuals ----
        #     while len(new_population) < pop_size:
        #         p1 = tournament_selection(population)
        #         p2 = tournament_selection(population)
        #
        #         c1, c2 = crossover(p1, p2, crossover_rate)
        #
        #         c1 = mutate(c1, mutation_rate)
        #         c2 = mutate(c2, mutation_rate)
        #
        #         new_population.append(c1)
        #         if len(new_population) < pop_size:
        #             new_population.append(c2)
        #
        #     population = new_population
        #
        #     # Track global best
        #     gen_best = min(population, key=fitness)
        #     if fitness(gen_best) < fitness(best):
        #         best = gen_best
        #
        #     print(f"Gen {gen:4d} | Best = {fitness(best):.3f}")
        #
        # return best
        pass


# ============================================================
#   4. Run GA
# ============================================================

if __name__ == "__main__":
    data = parse_txt('./Instance1.txt')
    print(list(data.keys()), data['staff'][0])
    solver = NurseRosteringGA(data)
    # solver.run()
    print(solver.pen_maximum_shift_types(solver.generate_individual()))
    # best_solution = genetic_algorithm()
    # print("\nBest Solution Found:")
    # print(best_solution)
    # print("Best Fitness:", fitness(best_solution))

