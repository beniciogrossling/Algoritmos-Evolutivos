#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io
import random
import json
import numpy
import time
import matplotlib.pyplot as plt
from deap import base, creator, algorithms, tools


# In[2]:


# Cargar datos desde el archivo JSON
with open("C:/Users/Benicio Grossling//Documents/Proyectos/Algoritmos Evolutivos/Capacited - VRPD/Algoritmos-Evolutivos/R101.json", "r") as archivo_json: 
    instance = json.load(archivo_json)


# In[3]:


def ind2route(individual, instance):
    route = []  # Inicializamos una lista vacía para almacenar las rutas
    vehicle_capacity = instance['vehicle_capacity']  # Capacidad máxima del vehículo
    max_drones = 2  # Máximo de clientes con cargas menores de 5 en cada sub-ruta

    # Inicializamos una sub-ruta vacía, así como las variables para el seguimiento del vehículo y los drones
    sub_route = []  # Sub-ruta actual
    vehicle_load = 0  # Carga actual del vehículo
    drone_load = 0  # Carga actual de drones
    last_customer_id = 0  # ID del último cliente visitado (comienza en 0, que es el depósito)

    # Recorremos el individuo (secuencia de clientes) uno por uno
    for customer_id in individual:
        # Obtenemos la demanda (carga) del cliente actual
        demand = instance[f'customer_{customer_id}']['demand']
        # Actualizamos la carga del vehículo si añadimos el cliente actual
        updated_vehicle_load = vehicle_load + demand
        # Validamos si la carga del vehículo no supera su capacidad máxima
        if (updated_vehicle_load <= vehicle_capacity): 
            # Validamos si aún no hemos alcanzado el máximo de drones
            if (demand <= 5) and (drone_load < max_drones):  # Si la carga es menor a 5, la asignamos a un dron
                drone_load = drone_load + 1
                # Si la carga es aceptable, agregamos el cliente a la sub-ruta actual
                sub_route.append(customer_id)
                vehicle_load = updated_vehicle_load  # Actualizamos la carga del vehículo
            elif (demand <= 5) and (drone_load == max_drones):
                continue
            else:
                sub_route.append(customer_id)
                vehicle_load = updated_vehicle_load  # Actualizamos la carga del vehículo
        else:
            # Si la carga supera la capacidad y ya hemos asignado el máximo de drones, guardamos la sub-ruta actual
            route.append(sub_route)
            # Inicializamos una nueva sub-ruta y agregamos el cliente actual a ella
            sub_route = [customer_id]
            vehicle_load = demand  # Reiniciamos la carga del vehículo con la del cliente actual
            drone_load = 0  # Reiniciamos la carga de drones

        # Actualizamos el ID del último cliente visitado
        last_customer_id = customer_id

    # Si todavía hay una sub-ruta no agregada, la guardamos antes de regresar
    if sub_route != []:
        route.append(sub_route)

    return route  # Devolvemos la lista de rutas resultante


def print_route(route, merge=False):
    # Inicializa una cadena para representar la ruta con el depósito inicial (0)
    route_str = '0'
    
    # Inicializa una variable para contar sub-rutas (vehículos)
    sub_route_count = 0
    
    # Itera a través de las sub-rutas en la ruta completa
    for sub_route in route:
        sub_route_count += 1  # Incrementa el contador de sub-rutas
        
        # Inicializa una cadena para representar la sub-ruta actual con el depósito inicial (0)
        sub_route_str = '0'
        
        # Inicializa una variable para rastrear la carga total de la sub-ruta
        sub_route_load = 0
        
        # Itera a través de los clientes en la sub-ruta actual
        for customer_id in sub_route:
            demand = instance[f'customer_{customer_id}']['demand']
            sub_route_str = f'{sub_route_str} - {customer_id} (Carga: {demand})'  # Agrega el cliente y su carga a la cadena de la sub-ruta
            # Suma la demanda (carga) del cliente a la carga total de la sub-ruta
            sub_route_load += demand
            
            # Agrega el cliente a la cadena de la ruta completa
            route_str = f'{route_str} - {customer_id}'
        
        sub_route_str = f'{sub_route_str} - 0'  # Agrega el depósito al final de la sub-ruta
        
        # Si el parámetro "merge" no está configurado como verdadero, imprime detalles de la sub-ruta
        if not merge:
            print(f'  Ruta del vehículo {sub_route_count}: {sub_route_str}')  # Imprime la sub-ruta con detalles de carga
            print(f'  Carga total de la sub-ruta: {sub_route_load}')  # Imprime la carga total de la sub-ruta
        
        # Agrega el depósito al final de la cadena de la ruta completa
        route_str = f'{route_str} - 0'
    
    # Si el parámetro "merge" está configurado como verdadero, imprime la ruta completa
    if merge:
        print(route_str)  # Imprime la ruta completa con todas las sub-rutas y depósitos


        
def plot_evolucion(log):
    gen = log.select("gen")
    fit_maxs = log.select("max")
    fit_ave = log.select("avg")
    fit_mins = log.select("min")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_maxs, label="Max Fitness")
    ax1.plot(gen, fit_ave, "--k", label="Avg Fitness")
    ax1.plot(gen, fit_mins, ":r", label="Min Fitness")
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    ax1.legend(loc="lower right")

    plt.show()       
        

        
def eval_vrp(individual, instance):
    total_cost = 0
    route = ind2route(individual, instance)
    for sub_route in route:
        sub_route_distance = 0
        last_customer_id = 0
        for customer_id in sub_route:
            demand = instance[f'customer_{customer_id}']['demand']
            
            # Exclude customers with demand less than 5 from distance calculation
            if demand <= 5:
                continue
            
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            
            # Update sub-route distance
            sub_route_distance += distance
            
            # Update last customer ID
            last_customer_id = customer_id
        
        # Calculate transport cost for the sub-route
        sub_route_distance += instance['distance_matrix'][last_customer_id][0]
        total_cost += sub_route_distance  # Suma el costo de esta subruta al costo total

    fitness = 1/total_cost  # Asigna el costo total como valor de aptitud
    return (fitness,)


def cx_partially_matched(ind1, ind2):
    cxpoint1, cxpoint2 = sorted(random.sample(range(min(len(ind1), len(ind2))), 2))
    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]
    rule1to2 = list(zip(part1, part2))
    is_fully_merged = False
    while not is_fully_merged:
        rule1to2, is_fully_merged = merge_rules(rules=rule1to2)
    rule2to1 = {rule[1]: rule[0] for rule in rule1to2}
    rule1to2 = dict(rule1to2)
    ind1 = [gene if gene not in part2 else rule2to1[gene] for gene in ind1[:cxpoint1]] + part2 + [gene if gene not in part2 else rule2to1[gene] for gene in ind1[cxpoint2+1:]]
    ind2 = [gene if gene not in part1 else rule1to2[gene] for gene in ind2[:cxpoint1]] + part1 + [gene if gene not in part1 else rule1to2[gene] for gene in ind2[cxpoint2+1:]]
    return (ind1, ind2)

def mut_inverse_indexes(individual):
    start, stop = sorted(random.sample(range(len(individual)), 2))
    temp = individual[start:stop+1]
    temp.reverse()
    individual[start:stop+1] = temp
    return (individual, )

def merge_rules(rules):
    is_fully_merged = True
    for round1 in rules:
        if round1[0] == round1[1]:
            rules.remove(round1)
            is_fully_merged = False
        else:
            for round2 in rules:
                if round2[0] == round1[1]:
                    rules.append((round1[0], round2[1]))
                    rules.remove(round1)
                    rules.remove(round2)
                    is_fully_merged = False
    return rules, is_fully_merged

def run_gavrp(instance_name, ind_size, pop_size, cx_pb, mut_pb, n_gen):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)

    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register('evaluate', eval_vrp, instance=instance)
    toolbox.register('select', tools.selTournament, tournsize=3)  # Cambiamos la selección a torneo
    toolbox.register('mate', cx_partially_matched)
    toolbox.register('mutate', mut_inverse_indexes)
    pop = toolbox.population(n=pop_size)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()

    # Results
    print('Start of evolution')

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f'  Evaluated {len(pop)} individuals')

    # Begin the evolution
    for gen in range(n_gen):
        print(f'-- Generation {gen} --')

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f'  Evaluated {len(invalid_ind)} individuals')

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x**2 for x in fits])
        std = abs(sum2 / length - mean**2)**0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')

        print('-- End of (successful) evolution --')
        best_ind = tools.selBest(pop, 1)[0]
        print(f'Best individual: {best_ind}')
        print(f'Fitness: {best_ind.fitness.values[0]}')
        print_route(ind2route(best_ind, instance))
        print(f'Total cost: {best_ind.fitness.values[0]}')

        # Guardar estadísticas en el logbook
        record = stats.compile(pop)
        logbook.record(gen=gen, min=min(fits), max=max(fits), avg=mean, std=std)

    return hof, logbook


# In[4]:


def main():
    
    # Se registra el tiempo de inicio
    start_time = time.time()
    
    random.seed(64)
    
    instance_name = 'R101.json'
    ind_size = 100
    pop_size = 400
    cx_pb = 0.85
    mut_pb = 0.02
    n_gen = 8000
    
    pop, logbook =  run_gavrp(instance_name=instance_name, ind_size=ind_size, pop_size=pop_size, cx_pb=cx_pb, mut_pb=mut_pb, n_gen=n_gen)
    
    # Mostrar la evolución del fitness
    plot_evolucion(logbook)

    # Se registra el tiempo de finalización
    end_time = time.time()

    # Calculo del tiempo transcurrido en segundos
    execution_time = end_time - start_time

    # Imprime el tiempo de ejecución
    print(f"Tiempo de ejecución: {execution_time} segundos")

    
    
    
if __name__ == '__main__':
    main()


# In[ ]:




