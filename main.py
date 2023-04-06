import streamlit as st
import numpy as np
from deap import algorithms, base, creator, tools
import time

# Define the constants
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
GENERATIONS = 10
NUM_CIRCLES = 20
WIDTH = 800
HEIGHT = 800

# Define the target square
TARGET_X = 400
TARGET_Y = 400
TARGET_SIZE = 50


# Define the fitness function
def fitness_function(individual):
    x = np.array(individual[::2])
    y = np.array(individual[1::2])
    return sum(1 / ((x - TARGET_X) ** 2 + (y - TARGET_Y) ** 2 + 1)),


# Define the genetic algorithm function
def genetic_algorithm(num_generations, num_circles):
    # Create the DEAP toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attribute", np.random.randint, 0, max(WIDTH, HEIGHT))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=num_circles * 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=max(WIDTH, HEIGHT), indpb=MUTATION_RATE)
    toolbox.register("select", tools.selTournament, tournsize=3, fit_attr="fitness")

    population = toolbox.population(n=POPULATION_SIZE)
    max_fitness = 0.0
    text_placeholder = st.empty()
    canvas_placeholder = st.empty()
    n_gens = num_generations

    while max_fitness < 1.0 and num_generations > 0:
        text_placeholder.text(f"Generation: {n_gens-num_generations}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        max_fitness = max([ind.fitness.values[0] for ind in population])

        # Set the colors
        CANVAS_COLOR = (255, 255, 224)  # Pastel yellow
        SQUARE_COLOR = (255, 200, 200)  # Pastel red
        CIRCLE_COLOR = (200, 200, 255)  # Pastel blue

        # Draw the circles on the canvas
        canvas = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        canvas[:] = CANVAS_COLOR
        canvas[TARGET_X - TARGET_SIZE // 2: TARGET_X + TARGET_SIZE // 2,
        TARGET_Y - TARGET_SIZE // 2: TARGET_Y + TARGET_SIZE // 2] = SQUARE_COLOR
        canvas[TARGET_X - TARGET_SIZE // 2 + 1: TARGET_X + TARGET_SIZE // 2 - 1,
        TARGET_Y - TARGET_SIZE // 2 + 1: TARGET_Y + TARGET_SIZE // 2 - 1] = SQUARE_COLOR

        for x, y in np.array(tools.selBest(population, k=1))[0].reshape(-1, 2):
            if (x >= TARGET_X - 5 and x <= TARGET_X + 5) and (y >= TARGET_Y - 5 and y <= TARGET_Y + 5):
                canvas[x - 5:x + 5, y - 5:y + 5] = (255, 200, 200)  # Pastel Red
            else:
                canvas[x - 5:x + 5, y - 5:y + 5] = CIRCLE_COLOR  # Pastel Blue

        canvas_placeholder.image(canvas, width=WIDTH)


# Define the main function
def main():
    st.title("Genetic Algorithm Visualizer")
    num_generations = st.slider("Number of Generations", 10, 1000, step=10)
    num_circles = st.slider("Number of Circles", 5, 50, step=5)
    genetic_algorithm(num_generations, num_circles)


# Call the main function
if __name__ == "__main__":
    main()
