from parse import parse_txt, parse_roster
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

    def clone(self):
        return ScheduleDay(self.shifts_limits, self.staff_limit, np.copy(self.gene))

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

    def mutate_swap(self):
        fp, sp = random.randint(0, len(self.gene)-1), random.randint(0, len(self.gene)-1)
        tmp = self.gene[fp]
        self.gene[fp] = self.gene[sp]
        self.gene[sp] = tmp

    def mutate_insert(self):
        l = [e for e in range(self.staff_limit) if e not in self.gene]
        if len(l) == 0: return
        new_worker = random.choice(l)
        fp = random.randint(0, len(self.gene)-1)
        self.gene[fp] = new_worker

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

    def clone(self):
        return Schedule(self.data, [ d.clone() for d in self.indiv])

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

PENAL_NUM = 8
class NurseRosteringGA:
    def __init__(self,
        problem_instance,
        pop_size=50,
        generations=200,
        crossover_rate=0.7,
        mutation_rate=0.05,
        elitism=1,
        penalities_weights=[2] * PENAL_NUM
    ):
        assert len(penalities_weights) == PENAL_NUM
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

    def pen_shift_rotation(self, info_table, s: Schedule):
        """
        An employee working consecutive shifts that are not allowed to follow each other
        """
        penalties = 0
        for i, cover in enumerate(self.data['cover'][:-1]):
            day = cover['day']
            shift = self.get_shift_from_id(cover['id'])
            if shift['cannot_follow']:
                for s in shift['cannot_follow']:
                    # verify if any worker is scheduled to work the shift 's' tomorrow
                    if s in info_table[day+1]:
                        intersection = info_table[day+1][s].intersection(info_table[day][shift['id']])
                        penalties += len(intersection)
        return penalties


    def pen_maximum_shift_types(self, info_table, s: Schedule):
        """
        An employee working more than the maximum allowed number of some shift type
        """
        penalties = 0
        for emp in self.data['staff']:
            limits = { s_id: {'cnt': 0, 'limit': int(limit_)} for (s_id, limit_) in emp['shift_limits'] }
            for day in range(len(s.indiv)):
                for (s_id, workers) in info_table[day].items():
                    if s_id == 'day_workers': continue
                    if self.employee_to_index[emp['id']] in workers:
                        limits[s_id]['cnt'] += 1

            for _, limit in limits.items():
                if limit['cnt'] > limit['limit']:
                    penalties += limit['cnt'] - limit['limit']

        return penalties

    def pen_min_max_working_time(self, info_table, s: Schedule):
        """
        An employee working less than the minimum or more than the maximum allowed working time
        """
        counter = np.zeros((self.staff_num,))
        for i, cover in enumerate(self.data['cover']):
            shift = self.get_shift_from_id(cover['id'])
            for worker in info_table[cover['day']][cover['id']]:
                counter[worker] += shift['len']
        penalties = 0
        for emp in self.data['staff']:
            emp_index = self.employee_to_index[emp['id']]
            min_time = emp['min_minu']
            max_time = emp['max_minu']

            # verify if the min or max time was violated
            if counter[emp_index] < min_time:
                penalties += min_time - counter[emp_index]
            if counter[emp_index] > max_time:
                penalties += counter[emp_index] - max_time
        return penalties

    def pen_min_max_consec_working_days_and_consec_days_off(self, info_table, s: Schedule) -> tuple[int, int, int]:
        """
        An employee working more than the maximum allowed number of consecutive working days
        returns (penalty_over, penalty_under, penalty_days_off)
        1st value: penalty for exceeding maximum consecutive working days
        2nd value: penalty for not reaching minimum consecutive working days
        3rd value: penalty for not reaching minimum consecutive days off
        """
        penal_over = 0
        penal_under = 0
        penal_days_off = 0
        for emp in self.data['staff']:
            max_consec = emp['max_cons_shifts']
            min_consec = emp['min_cons_shifts']
            min_consec_days_off = emp['min_cons_off']

            consec_count = 0
            days_off_count = 0
            for day in range(self.days_num):
                if emp['id'] in info_table[day]['day_workers']:
                    consec_count += 1
                    if 0 < days_off_count < min_consec_days_off:
                        penal_days_off += min_consec_days_off - days_off_count
                    days_off_count = 0
                else:
                    days_off_count += 1
                    if 0 < consec_count < min_consec:
                        penal_under += min_consec - consec_count
                    if consec_count > max_consec:
                        penal_over += consec_count - max_consec
                    consec_count = 0
        return penal_over, penal_under, penal_days_off

    def pen_working_weekends(self, info_table, s: Schedule):
        """
        An employee working more than the maximum allowed number of weekends
        """
        penalties = 0
        for emp in self.data['staff']:
            max_weekends = emp['max_weekends']
            weekend_count = 0
            for day in range(0, self.days_num, 7):
                # assuming weekend is day 5 and 6 of each week
                if (day + 5 < self.days_num and emp['id'] in info_table[day + 5]['day_workers']) \
                   or (day + 6 < self.days_num and emp['id'] in info_table[day + 6]['day_workers']):
                    weekend_count += 1
            if weekend_count > max_weekends:
                penalties += weekend_count - max_weekends
        return penalties

    def pen_working_on_days_off(self, info_table, s: Schedule):
        """
        An employee working on their days off
        """
        penalties = 0
        for emp_off in self.data['days_off']:
            for day_ in emp_off['days_off']:
                day = int(day_)
                if emp_off['staff_id'] in info_table[day]['day_workers']:
                    penalties += 1
        return penalties

    def constraint_penalty(self, info_table, s: Schedule):
        """
        Compute penalties for violating constraints.
        Return 0 if no violations.
        Increase value for worse constraint violations.
        """
        p, p2, p3 = self.pen_min_max_consec_working_days_and_consec_days_off(info_table, s)
        return self.penalities_weights[0] * self.pen_maximum_shift_types(info_table, s) + \
               self.penalities_weights[1] * self.pen_min_max_working_time(info_table, s) + \
               self.penalities_weights[2] * self.pen_shift_rotation(info_table, s) + \
               self.penalities_weights[3] * self.pen_working_on_days_off(info_table, s) + \
               self.penalities_weights[4] * self.pen_working_weekends(info_table, s) + \
               self.penalities_weights[5] * p + self.penalities_weights[6] * p2 + self.penalities_weights[7] * p3

    def objective_function(self, info_table, s: Schedule):
        """
        Compute the objective value.
        Lower is better (minimization).
        Modify this function with your real objective.
        """
        bad_approval = 0
        for req_on in self.data['requests']:
            if self.employee_to_index[req_on['staff_id']] not in info_table[req_on['day']][req_on['shift_id']]:
                bad_approval += req_on['weight']
        for req_off in self.data['requests_off']:
            if self.employee_to_index[req_off['staff_id']] in info_table[req_off['day']][req_off['shift_id']]:
                bad_approval += req_off['weight']
        return bad_approval

    def fitness(self, individual):
        """
        Final fitness = objective + penalties.
        """
        info_table = self.compute_indiv_info(individual)
        return self.objective_function(info_table, individual) + self.constraint_penalty(info_table, individual)


    # ============================================================
    #   2. Genetic Operators (Selection, Crossover, Mutation)
    # ============================================================

    def tournament_selection(self, population, k=3):
        """Pick best of k random individuals."""
        candidates = random.sample(population, k)
        return min(candidates, key=self.fitness)

    def crossover(self, p1: Schedule, p2):
        if random.random() > self.crossover_rate:
            if random.random() > 0.5:
                return p1.crossover_order_individual(p2)
            else:
                return p1.crossover_cycle_individual(p2)
        return p1.clone(), p2.clone()

    def mutate(self, s: Schedule):
        for day in s.indiv:
            if random.random() < self.mutation_rate:
                if random.random() > 0.5:
                    return day.mutate_insert()
                else:
                    return day.mutate_swap()

    def generate_neighbor(self, s: Schedule):
        """Gera um vizinho aplicando uma mutação local"""
        # TODO: Trocar por s.clone()
        neighbor = s.clone()
        #dia aleatório
        day = random.choice(neighbor.indiv)

        #tipo de movimento
        if random.random() < 0.5:
            day.mutate_swap()
        else:
            day.mutate_insert()

        return neighbor

    def hill_climbing(self, s: Schedule, max_iter=50):
        """Hill Climbing simples"""
        current = s
        current_fit = self.fitness(current)

        for _ in range(max_iter):
            neighbor = self.generate_neighbor(current)
            neighbor_fit = self.fitness(neighbor)

            if neighbor_fit < current_fit:
                current = neighbor
                current_fit = neighbor_fit
            else:
                break  #otimo local

        return current

    # ============================================================
    #   3. Main GA Loop
    # ============================================================

    def run(self):
        # ---- create initial population ----
        population = [self.generate_individual() for _ in range(self.pop_size)]
        new_population = []
        best = None

        for gen in range(self.generations):
            new_population.clear()

            # ---- elitism: keep the best individual ----
            elite = sorted(population, key=self.fitness)[:self.elitism]

            # ---- create new individuals ----
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)

                c1, c2 = self.crossover(p1, p2)

                self.mutate(c1)
                self.mutate(c2)

                c1 = self.hill_climbing(c1, max_iter=100)
                c2 = self.hill_climbing(c2, max_iter=100)

                new_population.append(c1)
                new_population.append(c2)

            final_pop = new_population + elite
            values = np.array(list(map(lambda x: self.fitness(x), final_pop)))
            probs = values / np.sum(values, dtype=float)
            population = list(np.random.choice(final_pop, size=self.pop_size, replace=False, p=probs))

            # Track global best
            best = min(population, key=self.fitness)
            # print(f"Gen {gen:4d} | Best = {self.fitness(best)} | Prev Best: {self.fitness(cur_best)}")
            # print(f"Gen {gen:4d} | Best = {self.fitness(best)}")
            print(f"{self.fitness(best)},", end=' ')

        return best

def solution_individual(solution_path, data):
    parsed = parse_roster(solution_path)
    nr = NurseRosteringGA(data)  # just to take the conversion between staff_id and index
    s = Schedule(data) # random individual to be overwritten
    for d in range(data['len_day']):
        s.indiv[d].gene = []
        s.indiv[d].shifts_limits.clear()

    penalties = 0
    for cover in data['cover']:
        workers = list(map(lambda y: nr.employee_to_index[y['staff_id']],
                filter(lambda x: x['shift_id'] == cover['id'] and x['day'] == cover['day'], parsed)))
        length = len(workers) 
        required = cover['requirement']
        if length != required:
            if length < required:
                penalties += cover['weight_under']
            else:
                penalties += cover['weight_over']
        s.indiv[cover['day']].gene.extend(workers)
        s.indiv[cover['day']].shifts_limits.append(length)

    # penalty weight for not scheduling the required amount of workers is 2.5
    return s, penalties * 2.5

# ============================================================
#   4. Run GA
# ============================================================

if __name__ == "__main__":
    data = parse_txt('./Instance4.txt')
    # for key, value in data.items():
    #     if isinstance(value, list) and len(value) > 2:
    #         print(f"{key}: {value[:2]}")
    #     else:
    #         print(f"{key}: {value}")
    solver = NurseRosteringGA(data,
        pop_size=50,
        generations=200,
        crossover_rate=0.7,
        mutation_rate=0.05,
        elitism=2,
    )
    solver.run()
    # best_solution = genetic_algorithm()
    # print("\nBest Solution Found:")
    # print(best_solution)
    # print("Best Fitness:", fitness(best_solution))

