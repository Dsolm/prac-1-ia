# This file contains all the required routines to make an A* search algorithm.
#
__author__ = '1708226'
# _________________________________________________________________________________________
# Intel.ligencia Artificial
# Curs 2023 - 2024
# Universitat Autonoma de Barcelona
# _______________________________________________________________________________________

from SubwayMap import *
from utils import *
import os
import math
import copy

def expand(path, map_p):
    connections = map(int, map_p.connections[path.last].keys())
    return [Path(path.route + [next]) for next in connections]

def remove_cycles(path_list):
    return [x for x in path_list if len(x.route) == len(set(x.route))]

def insert_depth_first_search(expand_paths, list_of_path):
    return expand_paths + list_of_path

def depth_first_search(origin_id, destination_id, map):
    llista = [Path(origin_id)]
    while llista and llista[0].last != destination_id:
        expanded = expand(llista[0], map)
        expanded = remove_cycles(expanded)
        llista = insert_depth_first_search(expanded, llista[1:])

    if llista:
        return llista[0]
    else:
        return llista

def insert_breadth_first_search(expand_paths, list_of_path):
    return list_of_path + expand_paths

def breadth_first_search(origin_id, destination_id, map):
    llista = [Path(origin_id)]
    while llista and llista[0].last != destination_id:
        expanded = expand(llista[0], map)
        expanded = remove_cycles(expanded)
        llista = insert_breadth_first_search(expanded, llista[1:])

    if llista:
        return llista[0]
    else:
        return llista

def distance_to_stations(coord, map):
    distancias = {id: euclidean_dist((vals["x"], vals["y"]), (coord[0], coord[1])) for id,vals in map.stations.items()}
    return dict(sorted(distancias.items(), key=lambda y: (y[1], y[0])))

def distance_to_stations(coord, map):
    dists = {id: euclidean_dist((vals["x"], vals["y"]), (coord[0], coord[1])) for id,vals in map.stations.items()}
    return dict(sorted(dists.items(), key=lambda x: (x[1], x[0])))

def distance_to_stations(coord, map):
    dist = {estacion_id: euclidean_dist((info['x'], info['y']), (coord[0], coord[1])) for estacion_id, info in map.stations.items()}
    dist_sorted = dict(sorted(dist.items(), key = lambda item: (item[1], item[0])))
    return dist_sorted


def calculate_cost(expand_paths, map, type_preference=0):
    def adjacency(expand_paths, map):
        for p in expand_paths:
            p.g = len(p.route) - 1
        return expand_paths

    def time(expand_paths, map):
        for path in expand_paths:
            distance = 0
            for i in range(1,len(path.route)):
                prev = path.route[i-1]
                now = path.route[i]
                time = map.connections[prev][now]
                distance += time
            path.g = distance
        return expand_paths

    def distance(expand_paths, map):
        for path in expand_paths:
            distance = 0
            for i in range(1,len(path.route)):
                prev = path.route[i-1]
                now = path.route[i]
                if map.stations[prev]["name"] != map.stations[now]["name"]:
                    line = map.stations[prev]["line"]
                    vel = map.velocity[line]
                    time = map.connections[prev][now]
                    distance += vel * time
                    path.g = distance
        return expand_paths

    def transfers(expand_paths, map):
        for path in expand_paths:
            transbordaments = 0
            for i in range(1,len(path.route)):
                prev = path.route[i-1]
                now = path.route[i]
                if map.stations[prev]["name"] == map.stations[now]["name"]:
                    transbordaments += 1
            path.g = transbordaments
        return expand_paths
    
    if not expand_paths:
        return []
    type = [adjacency, time, distance, transfers]
    return type[type_preference](expand_paths, map)


def insert_cost(expand_paths, list_of_path):
    copy = list_of_path + expand_paths
    return list(sorted(copy, key=lambda x: x.g))

def uniform_cost_search(origin_id, destination_id, map, type_preference=0):
    llista = [Path(origin_id)]
    while llista and llista[0].last != destination_id:
        expanded = expand(llista[0], map)

        expanded = remove_cycles(expanded)
        expanded = calculate_cost(expanded, map, type_preference)
        llista = insert_cost(expanded, llista[1:])

    if llista:
        return llista[0]
    else:
        return llista


def calculate_heuristics(expand_paths, map, destination_id, type_preference=0):
    def adjacency(expand_paths, map):
        for path in expand_paths:
            last = path.route[-1]
            if destination_id == last:
                path.update_h(0)
            else:
                path.update_h(1)
        return expand_paths

    def time(expand_paths, map):
        dest = map.stations[destination_id]
        max_speed = 0
        for path in expand_paths:
            for node in path.route:
                station = map.stations[node]
                if station["velocity"] > max_speed:
                    max_speed = station["velocity"]
            
        for path in expand_paths:
            last = path.route[-1]
            station = map.stations[last]
            distance = euclidean_dist((station["x"], station["y"]), (dest["x"], dest["y"]))/max_speed
            path.update_h(distance)
        return expand_paths

    def distance(expand_paths, map):
        dest = map.stations[destination_id]
        for path in expand_paths:
            last = path.route[-1]
            station = map.stations[last]
            vel = map.velocity[station["line"]]
            distance = euclidean_dist((station["x"], station["y"]), (dest["x"], dest["y"]))
            path.update_h(distance)
        return expand_paths

    def transfers(expand_paths, map):
        for path in expand_paths:
            before_last = path.route[-2]
            bl_station = map.stations[before_last]
            last = path.route[-1]
            l_station = map.stations[last]
            if bl_station["name"] == l_station["name"]:
                path.update_h(1)
            else:
                path.update_h(0)
        return expand_paths
    
    if not expand_paths:
        return []
    type = [adjacency, time, distance, transfers]
    return type[type_preference](expand_paths, map)



def update_f(expand_paths):
    """
      Update the f of a path
      Format of the parameter is:
         Args:
             expand_paths (LIST of Path Class): Expanded paths
         Returns:
             expand_paths (LIST of Path Class): Expanded paths with updated costs
    """
    for path in expand_paths:
        path.update_f()
    return expand_paths

def remove_redundant_paths(expand_paths, list_of_path, visited_stations_cost):
    for path in expand_paths:
        try:
            cp = visited_stations_cost[path.route[-1]]
            if path.g < cp:
                list_of_path = [x for x in list_of_path if x.route[-1] != path.route[-1] and x.g != cp]
                visited_stations_cost[path.route[-1]] = path.g
            else:
                expand_paths.remove(path)
        except KeyError:
            visited_stations_cost[path.route[-1]] = path.g
    return expand_paths, list_of_path, visited_stations_cost

class FKey:
    def __init__(self, path):
        self.path = path

    def __lt__(self, other):
        return self.path.f < other.path.f

def insert_cost_f(expand_paths, list_of_path):
    expand_paths = update_f(expand_paths)
    copy = list(list_of_path)
    for path in expand_paths:
        bisect.insort(copy, path, key=FKey)
    return copy

def calculate_heuristics(expand_paths, map, destination_id, type_preference=0):
    if type_preference == 0: # Boolean - Adjacent or not
        for path in expand_paths:
            path.update_h(1)
            if path.last == destination_id:
                path.update_h(0)      
        return expand_paths
    
    elif type_preference == 1: # Eucledian distance / max speed
        max_speed = 0
        for station_id, station_info in map.stations.items():
            velocity = station_info['velocity']
            if velocity > max_speed:
                max_speed = velocity
        dest_coor = [map.stations[destination_id]['x'], map.stations[destination_id]['y']]
        for path in expand_paths:
            if path.last in map.stations:
                coor = [map.stations[path.last]['x'], map.stations[path.last]['y']]
                path.update_h(euclidean_dist(coor, dest_coor) / max_speed)
        return expand_paths
    
    elif type_preference == 2: # Eucledian distance
        dest_coor = [map.stations[destination_id]['x'], map.stations[destination_id]['y']]
        for path in expand_paths:
            if path.last in map.stations:
                coor = [map.stations[path.last]['x'], map.stations[path.last]['y']]
                path.update_h(euclidean_dist(coor, dest_coor))
        return expand_paths
    
    elif type_preference == 3:  # Same line: h = 1; else: h = 0
        for path in expand_paths:
            if path.last == destination_id:
                path.update_h(0)
            else:
                line_number = map.stations[path.last]['line']
                dest_line_number = map.stations[destination_id]['line']
                if line_number != dest_line_number:
                    path.update_h(1)
        return expand_paths
    
    else:
        print("Invalid type_preference value")
        return 0


def Astar(origin_id, destination_id, map, type_preference=0):
    visited_stations_cost = dict()
    llista = [Path(origin_id)]
    while llista and llista[0].last != destination_id:
        expanded = expand(llista[0], map)
        expanded = remove_cycles(expanded)
        expanded = calculate_cost(expanded, map, type_preference)
        expanded,llista, visited_stations_cost = remove_redundant_paths(expanded, llista, visited_stations_cost)
        expanded = calculate_heuristics(expanded, map, destination_id, type_preference)
        llista = insert_cost_f(expanded, llista[1:])

    if llista:
        return llista[0]
    else:
        return llista

import numpy as np
from copy import deepcopy

def Astar_improved(origin_coord, destination_coord, _map):
    nmap = deepcopy(_map)
    # Crear estaciones de mentira para el inicio y el final caminando, la velocidad es la velocidad de caminar.
    nmap.stations.update({np.int64(0): {"name": "CAMINANT_INICI", "line": 0, "x": origin_coord[0], "y": origin_coord[1], "velocity": 5}})
    nmap.stations.update({np.int64(-1): {"name": "CAMINANT_FINAL", "line": 0, "x": destination_coord[0], "y": destination_coord[1], "velocity": 5} })
    # Hacer que el inicio caminando esté conectado con todas las estaciones
    nmap.connections.update(
        {np.int64(0):
         {id: np.float64(euclidean_dist((origin_coord[0], origin_coord[1]), (data["x"], data["y"]))/5) for id, data in nmap.stations.items()}
         }
    )
    # Añadir para cada estación una conexiónc on el destino caminando.
    for id, connection in nmap.connections.items():
        data = nmap.stations[id]
        connection[np.int64(-1)] = np.float64(euclidean_dist((destination_coord[0], destination_coord[1]), (data["x"], data["y"]))/5)
        
    type_preference = 1
    origin_id = 0
    destination_id = -1
    return Astar(origin_id, destination_id, nmap, type_preference)

def update_f(expand_paths):
    if len(expand_paths) >= 0:
        for path in expand_paths:
            path.update_f()
    return expand_paths


def insert_cost_f(expand_paths, list_of_path):
    list_of_path += expand_paths
    list_of_path.sort(key=lambda p: p.f)
    return list_of_path


def Astar(origin_id, destination_id, map, type_preference=0): 
    paths = [Path([origin_id])]
    visited_stations_cost = {}
    while len(paths) > 0 and paths[0].last != destination_id:
        path = paths.pop(0)
        expand_paths = expand(path, map)
        expand_paths = remove_cycles(expand_paths)
        expand_paths = calculate_cost(expand_paths, map, type_preference)
        expand_paths = calculate_heuristics(expand_paths, map, destination_id, type_preference)
        expand_paths = update_f(expand_paths)
        expand_paths, paths, visited_stations_cost = remove_redundant_paths(expand_paths, paths, visited_stations_cost)
        paths = insert_cost_f(expand_paths, paths)
    if not paths:
        return []
    elif paths[0].last == destination_id:
        return paths[0]

        

def Astar(origin_id, destination_id, map, type_preference=0):
    diccionario = dict()
    visitats = [Path(origin_id)]
    while visitats and visitats[0].last != destination_id:
        cap = visitats[0]
        llista = expand(cap, map)
        llista = remove_cycles(llista)
        llista = calculate_cost(llista, map, type_preference)
        llista,visitats,diccionario = remove_redundant_paths(llista, visitats, diccionario)
        llista = calculate_heuristics(llista, map, destination_id, type_preference)
        llista = update_f(llista)
        visitats = insert_cost_f(llista, visitats[1:])
    if visitats:
        return visitats[0]
    else:
        return []


def calculate_heuristics(path_expandidos, mapa, destination_id, type_preference=0):
    if type_preference == 0: 
        for p in path_expandidos:
            p.update_h(1)
            if p.last == destination_id:
                p.update_h(0)
                
    elif type_preference == 2: 
        for p in path_expandidos:
            if p.last in mapa.stations:
                c1 = (mapa.stations[p.last]['x'], mapa.stations[p.last]['y'])
                c2 = (mapa.stations[destination_id]['x'], mapa.stations[destination_id]['y'])
                h = euclidean_dist(c1, c2)
                p.update_h(h)
                                
    elif type_preference == 1: 
        vel_max = 0
        for estacion, info in mapa.stations.items():
            v = info['velocity']
            if v > vel_max:
                vel_max = v
        for p in path_expandidos:
            if p.last in mapa.stations:
                c1 = (mapa.stations[p.last]['x'], mapa.stations[p.last]['y'])
                c2 = (mapa.stations[destination_id]['x'], mapa.stations[destination_id]['y'])
                h = euclidean_dist(c1, c2) / vel_max
                p.update_h(h)
                
    elif type_preference == 3:  
        for p in path_expandidos:
            if p.last == destination_id:
                p.update_h(0)
            else:
                if mapa.stations[p.last]['line'] != mapa.stations[destination_id]['line']:
                    p.update_h(1)
                    
    return path_expandidos




def calculate_cost(expand_paths, map, type_preference=0):
    def type_temps(ep, map):
        for p in ep:
            max_temps = 0
            for i in range(1,len(p.route)):
                anterior = p.route[i-1]
                actual = p.route[i]
                time = map.connections[anterior][actual]
                max_temps += time
            p.g = max_temps
        return ep

    def type_adjacencia(ep, map):
        for p in ep:
            p.g = len(p.route) - 1
        return ep
    
    def type_transbords(ep, map):
        for p in ep:
            ts = 0
            for i in range(1,len(p.route)):
                anterior = p.route[i-1]
                actual = p.route[i]
                if map.stations[anterior]["name"] == map.stations[actual]["name"]:
                    ts += 1
            p.g = ts
        return expand_paths

    def type_dist(ep, map):
        for p in ep:
            dst = 0
            for i in range(1,len(p.route)):
                anterior = p.route[i-1]
                actual = p.route[i]
                if map.stations[anterior]["name"] != map.stations[actual]["name"]:
                    linea = map.stations[anterior]["line"]
                    dst += map.velocity[linea] * map.connections[anterior][actual]
                    p.g = dst
        return expand_paths
    
    type = [type_adjacencia, type_temps, type_dist, type_transbords]
    if not expand_paths:
        return []
    func = type[type_preference]
    return func(expand_paths, map)


def uniform_cost_search(origin, dest, map, type_preference=0):
    caminos_recorridos = [Path(origin)]
    while caminos_recorridos and caminos_recorridos[0].last != dest:
        camins_expandits = calculate_cost(remove_cycles(expand(caminos_recorridos[0], map)), map, type_preference)
        caminos_recorridos = insert_cost(camins_expandits, caminos_recorridos[1:])

    if caminos_recorridos:
        return caminos_recorridos[0]
    else:
        return caminos_recorridos


def insert_cost(expand_paths, list_of_path):
    copy = list_of_path + expand_paths
    return list(sorted(copy, key=lambda x: x.g))


def remove_redundant_paths(ep, lp, vsc):
    for path in ep:
        if path.route[-1] in vsc:
            cp = vsc[path.route[-1]]
            if path.g < cp:
                lp = [x for x in lp if x.route[-1] != path.route[-1] and x.g != cp]
                vsc[path.route[-1]] = path.g
            else:
                ep.remove(path)
        else:
            vsc[path.route[-1]] = path.g
    return ep, lp, vsc


def Astar(o_index, d_index, m, type_preference=0):
    table = dict()
    paths = [Path(o_index)]
    while paths and paths[0].last != d_index:
        cap = paths[0]
        expandido = remove_cycles(expand(cap, m))
        expandido = calculate_cost(expandido, m, type_preference)
        expandido,paths,table = remove_redundant_paths(expandido, paths, table)
        expandido = calculate_heuristics(expandido, m, d_index, type_preference)
        expandido = update_f(expandido)
        paths = insert_cost_f(expandido, paths[1:])
        
    if paths:
        return paths[0]
    else:
        return paths

def insert_cost_f(ep, lp):
    ep + lp
    
    for p in ep:
        lp.append(p)
    lp.sort(key=lambda p: p.f)
    return lp

def update_f(expanded):
    for path in expanded:
        path.update_f()
    return expanded

from copy import deepcopy
import numpy as np
def Astar_improved(origin_coord, destination_coord, _map):
    nmap = deepcopy(_map)
    estacion_inicio = {np.int64(0): {"name": "inicio_camino", "line": 0, "x": origin_coord[0], "y": origin_coord[1], "velocity": 5}}
    nmap.stations.update(estacion_inicio)
    estacion_final = {np.int64(-1): {"name": "FINAL_CAMINO", "line": 0, "x": destination_coord[0], "y": destination_coord[1], "velocity": 5} }
    nmap.stations.update(estacion_final)

    new_connections = {np.int64(0): {id: np.float64(euclidean_dist((origin_coord[0], origin_coord[1]), (data["x"], data["y"]))/5) for id, data in nmap.stations.items()}}
    nmap.connections.update(new_connections)
    
    for id, connection in nmap.connections.items():
        data = nmap.stations[id]
        connection[np.int64(-1)] = np.float64(euclidean_dist((destination_coord[0], destination_coord[1]), (data["x"], data["y"]))/5)
        
    type_preference = 1
    origin_id = 0
    destination_id = -1
    return Astar(origin_id, destination_id, nmap, type_preference)

def insert_cost_f(expand_paths, list_of_path):
    list_of_path += expand_paths
    list_of_path.sort(key=lambda p: p.f)
    return list_of_path


# haojie
def calculate_heuristics(path_expandidos, mapa, destination_id, type_preference=0):
    def heuristic_adj():
        for path in path_expandidos:
            path.update_h(0)
            if path.last != destination_id:
                path.update_h(1)

    def heuristic_dist():
        for path in path_expandidos:
            if path.last in mapa.stations:
                coord1 = [mapa.stations[path.last]['x'], mapa.stations[path.last]['y']]
                coord2 = [mapa.stations[destination_id]['x'], mapa.stations[destination_id]['y']]
                heur = euclidean_dist(coord1, coord2)
                path.update_h(heur)
                
    def heuristic_time():
        max = 0
        for estacion, info in mapa.stations.items():
            vel = info['velocity']
            if vel > max:
                max = vel
        for path in path_expandidos:
            if path.last in mapa.stations:
                coord2 = [mapa.stations[destination_id]['x'], mapa.stations[destination_id]['y']]
                coord1 = [mapa.stations[path.last]['x'], mapa.stations[path.last]['y']]
                heur = euclidean_dist(coord1, coord2) / max
                path.update_h(heur)
                
    def heuristic_trans():
        for path in path_expandidos:
            if path.last == destination_id:
                path.update_h(0)
            else:
                if mapa.stations[path.last]['line'] != mapa.stations[destination_id]['line']:
                    path.update_h(1)

    funs = [heuristic_adj, heuristic_time, heuristic_dist, heuristic_trans]
    fn = funs[type_preference]
    fn()
    return path_expandidos











import numpy as np
from copy import deepcopy

def expand(path, map_p):
    connections = map(int, map_p.connections[path.last].keys())
    return [Path(path.route + [next]) for next in connections]

def remove_cycles(path_list):
    return [x for x in path_list if len(x.route) == len(set(x.route))]

def calculate_cost(expand_paths, map, type_preference=0):
    def adjacency(expand_paths, map):
        for p in expand_paths:
            p.g = len(p.route) - 1
        return expand_paths

    def time(expand_paths, map):
        for path in expand_paths:
            distance = 0
            for i in range(1,len(path.route)):
                prev = path.route[i-1]
                now = path.route[i]
                time = map.connections[prev][now]
                distance += time
            path.g = distance
        return expand_paths

    def distance(expand_paths, map):
        for path in expand_paths:
            distance = 0
            for i in range(1,len(path.route)):
                prev = path.route[i-1]
                now = path.route[i]
                if map.stations[prev]["name"] != map.stations[now]["name"]:
                    line = map.stations[prev]["line"]
                    vel = map.velocity[line]
                    time = map.connections[prev][now]
                    distance += vel * time
                    path.g = distance
        return expand_paths

    def transfers(expand_paths, map):
        for path in expand_paths:
            transbordaments = 0
            for i in range(1,len(path.route)):
                prev = path.route[i-1]
                now = path.route[i]
                if map.stations[prev]["name"] == map.stations[now]["name"]:
                    transbordaments += 1
            path.g = transbordaments
        return expand_paths
    
    if not expand_paths:
        return []
    type = [adjacency, time, distance, transfers]
    return type[type_preference](expand_paths, map)

def insert_cost(expand_paths, list_of_path):
    copy = list_of_path + expand_paths
    return list(sorted(copy, key=lambda x: x.g))

def calculate_heuristics(path_expandidos, mapa, destination_id, type_preference=0):
    if type_preference == 0: 
        for p in path_expandidos:
            p.update_h(1)
            if p.last == destination_id:
                p.update_h(0)
                
    elif type_preference == 2: 
        for p in path_expandidos:
            if p.last in mapa.stations:
                c1 = (mapa.stations[p.last]['x'], mapa.stations[p.last]['y'])
                c2 = (mapa.stations[destination_id]['x'], mapa.stations[destination_id]['y'])
                h = euclidean_dist(c1, c2)
                p.update_h(h)
                                
    elif type_preference == 1: 
        vel_max = 0
        for estacion, info in mapa.stations.items():
            v = info['velocity']
            if v > vel_max:
                vel_max = v
        for p in path_expandidos:
            if p.last in mapa.stations:
                c1 = (mapa.stations[p.last]['x'], mapa.stations[p.last]['y'])
                c2 = (mapa.stations[destination_id]['x'], mapa.stations[destination_id]['y'])
                h = euclidean_dist(c1, c2) / vel_max
                p.update_h(h)
                
    elif type_preference == 3:  
        for p in path_expandidos:
            if p.last == destination_id:
                p.update_h(0)
            else:
                if mapa.stations[p.last]['line'] != mapa.stations[destination_id]['line']:
                    p.update_h(1)
                    
    return path_expandidos

def remove_redundant_paths(expand_paths, list_of_path, visited_stations_cost):
    for path in expand_paths:
        try:
            cp = visited_stations_cost[path.route[-1]]
            if path.g < cp:
                list_of_path = [x for x in list_of_path if x.route[-1] != path.route[-1] and x.g != cp]
                visited_stations_cost[path.route[-1]] = path.g
            else:
                expand_paths.remove(path)
        except KeyError:
            visited_stations_cost[path.route[-1]] = path.g
    return expand_paths, list_of_path, visited_stations_cost




def Astar(o_index, d_index, m, type_preference=0):
    table = dict()
    paths = [Path(o_index)]
    while paths and paths[0].last != d_index:
        cap = paths[0]
        expandido = remove_cycles(expand(cap, m))
        expandido = calculate_cost(expandido, m, type_preference)
        expandido,paths,table = remove_redundant_paths(expandido, paths, table)
        expandido = calculate_heuristics(expandido, m, d_index, type_preference)
        expandido = update_f(expandido)
        paths = insert_cost_f(expandido, paths[1:])
    if paths:
        return paths[0]
    else:
        return paths

def update_f(expanded):
    for path in expanded:
        path.update_f()
    return expanded

def insert_cost_f(expand_paths, list_of_path):
    list_of_path += expand_paths
    list_of_path.sort(key=lambda p: p.f)
    return list_of_path

def Astar_improved(origin_coord, destination_coord, _map):
    nmap = deepcopy(_map)
    nmap.stations.update({np.int64(0): {"name": "CAMINANT_INICI", "line": 0, "x": origin_coord[0], "y": origin_coord[1], "velocity": 5}})
    nmap.stations.update({np.int64(-1): {"name": "CAMINANT_FINAL", "line": 0, "x": destination_coord[0], "y": destination_coord[1], "velocity": 5} })
    nmap.connections.update(
        {np.int64(0):
         {id: np.float64(euclidean_dist((origin_coord[0], origin_coord[1]), (data["x"], data["y"]))/5) for id, data in nmap.stations.items()}
         }
    )
    for id, connection in nmap.connections.items():
        data = nmap.stations[id]
        connection[np.int64(-1)] = np.float64(euclidean_dist((destination_coord[0], destination_coord[1]), (data["x"], data["y"]))/5)
        
    type_preference = 1
    origin_id = 0
    destination_id = -1
    return Astar(origin_id, destination_id, nmap, type_preference)
