from SearchAlgorithm import *
from SubwayMap import *
from utils import *
import unittest
import os
import numpy as np

def load_map():
        
    ROOT_FOLDER = './CityInformation/Barcelona_City/'
    map = read_station_information(os.path.join(ROOT_FOLDER, 'Stations.txt'))
    connections = read_cost_table(os.path.join(ROOT_FOLDER, 'Time.txt'))
    map.add_connection(connections)

    infoVelocity_clean = read_information(os.path.join(ROOT_FOLDER, 'InfoVelocity.txt'))
    map.add_velocity(infoVelocity_clean)

    ### BELOW HERE YOU CAN CALL ANY FUNCTION THAT yoU HAVE PROGRAMED TO ANSWER THE QUESTIONS OF THE EXAM ###
    ### this code is just for you, you won't have to upload it after the exam ###

    #this is an example of how to call some of the functions that you have programed
    #example_path=uniform_cost_search(9, 3, map, 1)
    #print_list_of_path_with_cost([example_path])

    #b = [Path([1,2,3,4])]
    #x = calculate_cost(b, map, 2)
    #print_list_of_path_with_cost(x)

    #x=breadth_first_search(19,14,map)
    #print_list_of_path_with_cost([x])

    """path=[Path([5,6,3]),Path([5,6,2]),Path([5,6,1]),Path([5,6,8,4])]
    cost=[Path([17,30,54,46])]
    expath = [Path([5, 6, 8,3]), Path([5, 6, 8,2]), Path([5, 6, 8,1]), Path([5, 6, 8, 7])]
    costu={6:5.85,8:3.11,3:21.46,2:35.9,1:24.9}
    x=remove_redundant_paths(path,expath,costu)
    print_list_of_path(x)

    y=calculate_heuristics(path,map,13,1)
    print_list_of_path_with_cost(y)"""

    return map

def exemple():
    '''1
    使用 type_preference = 2（欧几里得距离）
    路径只含 [8,9]，计算 9 -> 13 的启发式 h
    '''
    pathbb=[Path([8,9])]
    print(calculate_heuristics(pathbb,map,13,2)[0].h)
    '''2
    这是测试将任意坐标映射到最近车站的排序。
    '''
    #s=distance_to_stations([107,150],map)
    #print(s)
    '''3
    g: 由时间计算
    h: 由欧几里得/时间估计到 13
    输出 f = g + h
    '''
    list_of_path=[Path([8,9,10,23])]
    a=remove_cycles(list_of_path)
    x=calculate_cost(a,map,1)
    y=calculate_heuristics(x,map,13,1)
    expandded=update_f(y)
    print(expandded[0].f)
    '''4
    这是标准 A* 路径求解题，偏好为时间。
    '''
    A=Astar(23,13,map,1)
    print([A][0].f)
    print_list_of_path_with_cost([A])
    '''5
    测试是否新的路径比已有的便宜，会替换旧路径
    '''
    #expand_paths=[Path([3,1,7,2])]
    #cost_dict = distance_to_stations([69,138],map)
    #remove_redundant_paths(expand_paths,list_of_path,cost_dict)
    '''6
    这个测试意图是验证路径去环功能，特别是像 [5,3,2,1,5]、[5,3,2,1,3] 这种有环的
    '''
    path=[Path([5,3,2,1,6]),Path([5,3,2,1,5]),Path([5,3,2,1,3]),Path([5,3,2,1,8])]
    '''7
    这也是 A* 路径测试题。
    '''
    #a=Astar(9,23,map,1)
    #print_list_of_path_with_cost([a])

def exemple2():
    
    #2
    #这测试的是 从任意位置找到最近地铁站，然后打印该站信息（站点 17 是 Ciutadella-Vila Olimpica）
    #a = distance_to_stations([0,223], map)
    #print(map.stations[17])

    #3
    '''
    [6,1,7,3,1] 和 [6,1,7,3,8] 中可能存在回环或多次访问点；
    这题考你是否能识别并移除路径中的“回头路”。
    list_path = [Path([2, 7, 8, 3, 5])]
    list_path.append(Path([2, 7, 8, 3, 4]))
    list_path.append(Path([2, 7, 8, 3, 2]))
    list_path.append(Path([2, 7, 8, 3, 7]))
    list_path.append(Path([2, 7, 8, 3, 1]))
    print_list_of_path_with_cost(remove_cycles(list_path))
    '''

    #4
    '''    
    这是 BFS 路径计算题，最短步数优先。
    a = breadth_first_search(22, 17, map)
    print_list_of_path_with_cost([a])
    '''
    #5
    #跟 Pregunta 2 类似，但这次是另一条路径；
    #输出的 g 值是重点（distance = velocity × time）
    #a = calculate_cost([Path([10,23,22,21,20,19,18,11,12,13])], map, 2)

    #6
    #这题只计算启发值 h，时间启发（欧几里得距离 / 45）
    #a = calculate_heuristics([Path(8)], map, 12, 1)

    #7
    '''   
    前三段是实际代价
    最后一段是启发代价,使用最快线路的速度估计
    a = map.connections[13][14]
    a += map.connections[14][15]
    a += map.connections[15][16]
    d = euclidean_dist((map.stations[16]['x'], map.stations[16]['y']),
                   (map.stations[18]['x'], map.stations[18]['y']))
    a += d / max(map.velocity.values())
    '''


def pregunta1():
    map = load_map()
    origin_id = 21  # Clot L4
    destination_id = 16  # Bogatell L3

    path = breadth_first_search(origin_id, destination_id, map)
    
    if path:
        print("Pregunta 1: Camí de Clot L4 a Bogatell L3 (BFS)")
        print("Resultat IDs:", path.route)
        
        options = {
            'A': [21, 20, 19, 14, 18, 11, 12, 13, 14, 15, 16],
            'B': [21, 20, 5, 19, 18, 11, 12, 13, 14, 15, 16],
            'C': [21, 20, 19, 18, 11, 12, 13, 14, 15, 16],
            'D': [21, 20, 19, 12, 18, 11, 12, 13, 14, 15, 16],
        }

        for k, v in options.items():
            if path.route == v:
                print(f"✔️ Resposta correcta: {k}")
                break
        else:
            print("❌ Cap opció coincideix exactament amb la ruta trobada.")
    else:
        print("No s'ha trobat cap camí")
    return None

def pregunta2():
    map = load_map()

    # 构建一个 Path 对象，包含完整路径
    path = Path([10, 23, 22, 21, 20, 19, 18, 11, 12, 13])
    
    # 放入列表以便调用 calculate_cost
    path_list = [path]

    # 使用 type_preference=2 计算距离代价
    path_list = calculate_cost(path_list, map, 2)

    # 输出结果
    print("Pregunta: Quin serà el cost(G) en Distància del camí: [5, 4, 3, 2, 1, 7, 8]?")
    print(f"Ruta: {path.route}")
    print(f"Cost G (Distància): {path_list[0].g}")

def pregunta_distancia_G():
    map = load_map()

    # 构建一个 Path 对象，包含完整路径
    path = Path([10, 23, 22, 21, 20, 19, 18, 11, 12, 13])
    
    # 放入列表以便调用 calculate_cost
    path_list = [path]

    # 使用 type_preference=2 计算距离代价
    path_list = calculate_cost(path_list, map, type_preference=2)

    # 输出结果
    print("Pregunta: Quin serà el cost(G) en Distància del camí: [5, 4, 3, 2, 1, 7, 8]?")
    print(f"Ruta: {path.route}")
    print(f"Cost G (Distància): {path.g}")


def problema1():
    map = load_map()
    coord = [134, 192]
    distancies = distance_to_stations(coord, map)
    estacio_id_propera = list(distancies.keys())[0]
    estacio_info = map.stations[estacio_id_propera]
    print(f"Estació ID: {estacio_id_propera}")
    print(f"Nom: {estacio_info['name']}")
    print(f"Línia: {estacio_info['line']}")


def problema2():
    map = load_map()
    list_path = [Path([2, 7, 8, 3, 5])]
    list_path.append(Path([2, 7, 8, 3, 4]))
    list_path.append(Path([2, 7, 8, 3, 2]))
    list_path.append(Path([2, 7, 8, 3, 7]))
    list_path.append(Path([2, 7, 8, 3, 1]))
    print_list_of_path_with_cost(remove_cycles(list_path))

def problema3():
    map = load_map()
    a = breadth_first_search(15, 10, map)
    print_list_of_path_with_cost([a])

def pregunta_cost_distancia_llarga():
    map = load_map()
    ruta = [10, 23, 22, 21, 20, 19, 18, 11, 12, 13]
    test_path = Path(ruta)
    result = calculate_cost([test_path], map, type_preference=2)
    print(f"Cost G = {result[0].g:.2f}")

def pregunta_heuristica_temps():
    map = load_map()
    estacio_actual = 9   
    destinacio = 15     
    coord_actual = (map.stations[estacio_actual]['x'], map.stations[estacio_actual]['y'])
    coord_dest = (map.stations[destinacio]['x'], map.stations[destinacio]['y'])
    distancia = euclidean_dist(coord_actual, coord_dest)
    velocitat_max = max(map.velocity.values())
    h = distancia / velocitat_max
    print(f"h = {h:.2f}")

def pregunta_7_f_temps():
    map = load_map()
    ruta = [15, 14, 13, 12]
    dest = 20  
    path = Path(ruta)
    calculate_cost([path], map, type_preference=1) 
    calculate_heuristics([path], map, dest, type_preference=1)
    path.update_f()
    print(f" F = {path.f:.2f}")


def problema_9_astar():
    map = load_map()
    a=Astar(15,3,map,1)
    print_list_of_path_with_cost([a])


def pregunta_astar_temps():
    map = load_map()
    path = Astar(5, 9, map, type_preference=1)
    if path:
        print(f"Cost G: {path.g:.2f}")

if __name__=="__main__":
    problema3()
    problema2()
    problema1()