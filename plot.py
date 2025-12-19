
def get_fitness(file_path):
    with open(file_path, 'r') as f:
        s = list(map(lambda x: int(float(x.strip())), [x for x in f.readlines()[0].split(',') if x.strip() ]))
        return s

import matplotlib.pyplot as plt

# Example data: best fitness per generation

best_fitness_model_1 = get_fitness('./i4_outputs/i4_ps=50_g=100_normal')[5:]
best_fitness_model_2 = get_fitness('./i4_outputs/i4_ps=200_g=100_normal')[5:]
best_fitness_model_3 = get_fitness('./i4_outputs/i4_ps=100_g=200_normal')[5:]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(len(best_fitness_model_1)), best_fitness_model_1, label="Pop. size = 50 and 100 generations")
plt.plot(range(len(best_fitness_model_2)), best_fitness_model_2, label="Pop. size = 200 and 100 generations")
plt.plot(range(len(best_fitness_model_3)), best_fitness_model_3, label="Pop. size = 100 and 200 generations")

# Labels and title
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Best Fitness per Generation")
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.savefig('plot.png')

get_fitness('./i5_ps=50_g=100_normal')