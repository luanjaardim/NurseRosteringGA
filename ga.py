from parse import parse_txt
import random
import numpy as np
random.seed(42)
np.random.seed(42)

class ScheduleDay:
    def __init__(self, shifts_requirements : list[int], staff_limit, gene=None):
        self.shifts_limits = shifts_requirements
        self.workers_num = sum(shifts_requirements)
        self.staff_limit = staff_limit
        self.shifts_num = len(shifts_requirements)
        assert(self.workers_num <= self.staff_limit)
        if gene is None:
            self.gene = np.random.permutation(self.staff_limit)[:self.workers_num]
        else:
            assert(sum(shifts_requirements) == len(gene))
            self.gene = gene

    def __str__(self):
        return f'(ScheduleDay {self.shifts_limits}) : {self.gene}'

    def crossover_cycle_day(self, other):
        p1 = self.gene
        p2 = other.gene
        size = len(p1)

        #Corrigir valores faltantes
        set1 = set(p1)
        set2 = set(p2)

        #valores que faltam em cada pai
        faltando_em_p2 = list(set1 - set2)

        #criar listas auxiliares baseadas nos pais reais
        p2_cycle = p2.copy()

        #substituir valores em p2 para incluir todos os valores de p1
        j = 0
        for i in range(size):
            if p2_cycle[i] not in set1:
                p2_cycle[i] = faltando_em_p2[j]
                j += 1

        c1 = np.full(size, -1)
        c2 = np.full(size, -1)

        visited = set()
        cycle_index = 0

        #Vai percorrer os ciclos ate alcaçar a aquantidade
        while len(visited) < size:

            #Encontra a próxima posição ainda não visitado
            start = next(i for i in range(size) if i not in visited)
            index = start
            cycle = []

            #Construir ciclo ate que o index ja tenha sido visitado
            while index not in visited:
                cycle.append(index)
                visited.add(index)

                #valor em P2 na posição atual
                #value = p2[index] #Se a gente mexer na 
                value = p2_cycle[index]

                #próxima posição = onde P1 tem esse mesmo valor
                index = np.where(p1 == value)[0][0]

            #Verificando qual a geração atual com o impar ou par para alterna a alimentação dos genes dos pais para os filhos
            if cycle_index % 2 == 0:
                #Ciclo par => P1 => C1
                for i in cycle:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                #Ciclo ímpar => P2 => C1
                for i in cycle:
                    c1[i] = p2[i]
                    c2[i] = p1[i]

            cycle_index += 1

        child1 = ScheduleDay(self.shifts_limits, self.staff_limit, c1)
        child2 = ScheduleDay(other.shifts_limits, other.staff_limit, c2)

        return child1, child2

    def crossover_order_day(self, other):

            size = len(self.gene) #p1_day.shifts_limits
            c1_gene = np.full(size, -1)
            c2_gene = np.full(size, -1)

            #Definir intervalo [lower, upper)
            lower = random.randint(0, size - 2)
            upper = random.randint(lower + 1, size - 1)

            #Copiar o bloco do pai correspondente para cada filho
            c1_gene[lower:upper] = self.gene[lower:upper]
            c2_gene[lower:upper] = other.gene[lower:upper]

            #Posições vazias
            fill_pos_c1 = [i for i in range(size) if c1_gene[i] == -1]
            fill_pos_c2 = [i for i in range(size) if c2_gene[i] == -1]

            #Genes restantes preservando ordem, verificar nos filhos quais trabalhadores ja foram atribuidos nos genes
            p2_list = [g for g in other.gene if g not in c1_gene]
            p1_list = [g for g in self.gene if g not in c2_gene]

            for pi1, pos1 in enumerate(fill_pos_c1):
                c1_gene[pos1] = p2_list[pi1]

            for pi2, pos2 in enumerate(fill_pos_c2):
                c2_gene[pos2] = p1_list[pi2]

            c1 = ScheduleDay(self.shifts_limits, self.staff_limit, gene=c1_gene)
            c2 = ScheduleDay(other.shifts_limits, other.staff_limit, gene=c2_gene)

            return c1, c2

class Schedule:
    def __init__(self, data, indiv=None):
        """ Create a random candidate solution. """
        self.data = data
        if indiv is None:
            day = 0
            shifts_lengths = []
            self.indiv = []
            staff_limit = len(data['staff'])
            for cover in data['cover']:
                if cover['day'] == day:
                    shifts_lengths.append(cover['requirement'])
                else:
                    day += 1
                    self.indiv.append(ScheduleDay(shifts_lengths, staff_limit))
                    shifts_lengths = [cover['requirement']]
            self.indiv.append(ScheduleDay(shifts_lengths, staff_limit))
        else:
            self.indiv = indiv
            assert(len(self.indiv) == data['len_day'])
            assert(all(map(lambda x: isinstance(x, ScheduleDay), self.indiv)))

    def __str__(self):
        return f'(Schedule): {[str(sday) for sday in self.indiv]}'

    def print(self):
        print("(Schedule): [")
        for day in self.indiv:
            print(f"\t{str(day)}")
        print("]")

    def crossover_cycle_individual(self, other):
        """Faz OX dia a dia e cria um novo indivíduo"""
        c1_days = []
        c2_days = []

        #ambos têm o mesmo número de dias
        for day_idx in range(len(self.indiv)):
            p1_day = self.indiv[day_idx]
            p2_day = other.indiv[day_idx]

            c1_day, c2_day = p1_day.crossover_cycle_day(p2_day)

            c1_days.append(c1_day)
            c2_days.append(c2_day)

        #cria novos individuos
        child1 = Schedule(self.data, c1_days)
        child2 = Schedule(other.data, c2_days)

        return child1, child2

    def crossover_order_individual(self, other):
        c1_days = []
        c2_days = []

        #ambos têm o mesmo número de dias
        for day_idx in range(len(self.indiv)):
            p1_day = self.indiv[day_idx]
            p2_day = other.indiv[day_idx]

            c1_day, c2_day = p1_day.crossover_order_day(p2_day)

            c1_days.append(c1_day)
            c2_days.append(c2_day)

        #cria novos individuos
        child1 = Schedule(self.data, c1_days)
        child2 = Schedule(self.data, c2_days)
        return child1, child2

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
        self.data = problem_instance
        self.staff_num = len(problem_instance['staff'])
        self.days_num = problem_instance['len_day']
        # Number of covers to choose a staff to
        self.individual_size = len(problem_instance['cover'])
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
        return self.data['shifts'][index]

    def get_employee_from_id(self, id):
        index = self.employee_to_index[id]
        return self.data['staff'][index]

    def generate_individual(self):
        """ Create a random candidate solution. """
        return Schedule(self.data)

    def compute_indiv_info(self, s: Schedule):
        info_table = [
            dict() for _ in range(len(s.indiv))
        ]
        day = -1
        prev_ind_gene = 0
        s_ind = 0
        for cover in self.data['cover']:
            if day != cover['day']:
                s_ind = 0
                prev_ind_gene = 0
                info_table[day]['day_workers'] = set(s.indiv[day].gene)
                day = cover['day']
            else:
                s_ind += 1

            end_shift_workers = s.indiv[day].shifts_limits[s_ind]
            info_table[day][cover['id']] = set(s.indiv[day].gene[prev_ind_gene:prev_ind_gene + end_shift_workers])
            prev_ind_gene += end_shift_workers
        return info_table

    def objective_function(self, indiv):
        """
        Compute the objective value.
        Lower is better (minimization).
        Modify this function with your real objective.
        """
        pass

    def pen_shift_rotation(self, indiv):
        """
        An employee working consecutive shifts that are not allowed to follow each other
        """
        mat_shift_types = []
        for _ in range(self.staff_num): mat_shift_types.append([set() for _ in range(self.days_num)])
        for i, cover in enumerate(self.instance_data['cover']):
            # each element of this matrix is a set of the shifts of an employee at some day
            mat_shift_types[indiv[i]][cover['day']].add(cover['id'])

        penalties = 0
        for i, cover in enumerate(self.instance_data['cover'][1:], start=1):
            day = cover['day']
            shift = self.get_shift_from_id(cover['id'])
            if shift['cannot_follow']:
                for s in shift['cannot_follow']:
                    # verify if the worker worked this same shift 's' on the previous day
                    if s in mat_shift_types[indiv[i]][day-1]:
                        penalties += 1
        return penalties


    def pen_maximum_shift_types(self, indiv):
        """
        An employee working more than the maximum allowed number of some shift type
        """
        counter = np.zeros((self.staff_num, len(self.instance_data['shifts']),))
        for i, cover in enumerate(self.instance_data['cover']):
            # counts the number of each shift type for each worker
            counter[indiv[i]][self.shift_to_index[cover['id']]] += 1
        penalties = 0
        for emp in self.instance_data['staff']:
            for (s_id, limit_) in emp['shift_limits']:
                limit = int(limit_)
                emp_index = self.employee_to_index[emp['id']]
                shi_index = self.shift_to_index[s_id]

                # verify if the limit was reached
                if counter[emp_index][shi_index] > limit:
                    penalties += counter[emp_index][shi_index] - limit
        return penalties

    def pen_min_max_working_time(self, indiv):
        """
        An employee working less than the minimum or more than the maximum allowed working time
        """
        counter = np.zeros((self.staff_num,))
        for i, cover in enumerate(self.instance_data['cover']):
            shift = self.get_shift_from_id(cover['id'])
            counter[indiv[i]] += shift['len']
        penalties = 0
        for emp in self.instance_data['staff']:
            emp_index = self.employee_to_index[emp['id']]
            min_time = emp['min_minu']
            max_time = emp['max_minu']

            # verify if the min or max time was violated
            if counter[emp_index] < min_time:
                penalties += min_time - counter[emp_index]
            if counter[emp_index] > max_time:
                penalties += counter[emp_index] - max_time
        return penalties

    def pen_min_max_consec_working_days_and_consec_days_off(self, indiv) -> tuple[int, int, int]:
        """
        An employee working more than the maximum allowed number of consecutive working days
        returns (penalty_over, penalty_under, penalty_days_off)
        1st value: penalty for exceeding maximum consecutive working days
        2nd value: penalty for not reaching minimum consecutive working days
        3rd value: penalty for not reaching minimum consecutive days off
        """
        mat_working_days = np.zeros((self.staff_num, self.days_num,))
        for i, cover in enumerate(self.instance_data['cover']):
            mat_working_days[indiv[i]][cover['day']] = 1

        penal_over = 0
        penal_under = 0
        penal_days_off = 0
        for emp in self.instance_data['staff']:
            emp_index = self.employee_to_index[emp['id']]
            max_consec = emp['max_cons_shifts']
            min_consec = emp['min_cons_shifts']
            min_consec_days_off = emp['min_cons_off']

            consec_count = 0
            days_off_count = 0
            for day in range(self.days_num):
                if mat_working_days[emp_index][day] == 1:
                    consec_count += 1
                    if consec_count > max_consec:
                        penal_over += consec_count - max_consec
                    if 0 < days_off_count < min_consec_days_off:
                        penal_days_off += min_consec_days_off - days_off_count
                    days_off_count = 0
                else:
                    days_off_count += 1
                    if 0 < consec_count < min_consec:
                        penal_under += min_consec - consec_count
                    consec_count = 0
        return penal_over, penal_under

    def pen_working_weekends(self, indiv):
        """
        An employee working more than the maximum allowed number of weekends
        """
        mat_working_days = np.zeros((self.staff_num, self.days_num,))
        for i, cover in enumerate(self.instance_data['cover']):
            mat_working_days[indiv[i]][cover['day']] = 1

        penalties = 0
        for emp in self.instance_data['staff']:
            emp_index = self.employee_to_index[emp['id']]
            max_weekends = emp['max_weekends']

            weekend_count = 0
            for day in range(0, self.days_num, 7):
                # assuming weekend is day 5 and 6 of each week
                if (day + 5 < self.days_num and mat_working_days[emp_index][day + 5] == 1) \
                   or (day + 6 < self.days_num and mat_working_days[emp_index][day + 6] == 1):
                    weekend_count += 1
            if weekend_count > max_weekends:
                penalties += weekend_count - max_weekends
        return penalties

    def pen_working_on_days_off(self, indiv):
        """
        An employee working on their days off
        """
        mat_days_off = np.zeros((self.staff_num, self.days_num,))
        for i, emp_off in enumerate(self.instance_data['days_off']):
            emp_index = self.employee_to_index[emp_off['staff_id']]
            for d in emp_off['days_off']:
                mat_days_off[emp_index][d] = 1

        penalties = 0
        for i, cover in enumerate(self.instance_data['cover']):
            day = cover['day']
            emp_index = indiv[i]
            if mat_days_off[emp_index][day] == 1:
                penalties += 1
        return penalties

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
        population = [str(self.generate_individual()) for _ in range(1)]
        print(population)
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
    data = parse_txt('./Instance4.txt')
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 2:
            print(f"{key}: {value[:2]}")
        else:
            print(f"{key}: {value}")
    solver = NurseRosteringGA(data)
    solver.run()
    # best_solution = genetic_algorithm()
    # print("\nBest Solution Found:")
    # print(best_solution)
    # print("Best Fitness:", fitness(best_solution))

