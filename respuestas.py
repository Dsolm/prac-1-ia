from SearchAlgorithm import *
from SubwayMap import *
from utils import *
import pprint

def load_map() -> Map:
    ROOT_FOLDER = 'CityInformation/Barcelona_City/'
    map = read_station_information(os.path.join(ROOT_FOLDER, 'Stations.txt'))
    connections = read_cost_table(os.path.join(ROOT_FOLDER, 'Time.txt'))
    map.add_connection(connections)

    infoVelocity_clean = read_information(os.path.join(ROOT_FOLDER, 'InfoVelocity.txt'))
    map.add_velocity(infoVelocity_clean)

    return map

map = load_map()
path = breadth_first_search(21, 16, map)

# 2.
dist = distance_to_stations([181, 197], map)
num_dist = dist.items()
dist_num = [(d,n) for n,d in num_dist]
station = min(dist_num)[1]
station_name = map.stations[station]
print("1. ", station_name)

# 3.
listas = [[7, 6, 1, 8, 5], [7, 6, 1, 8, 2], [7, 6, 1, 8, 7], [7, 6, 1, 8, 6], [7, 6, 1, 8, 4]]
paths = [Path(route) for route in listas]
paths = remove_cycles(paths)
print("2. ", [p.route for p in paths])

# 4.
sagregal2 = 7
navasl1 = 2
path = breadth_first_search( sagreral2, navasl1, map)
print(path.route)

# 5.
print("5. ", calculate_cost([Path([18, 19, 20, 21])], map, 2)[0].g)

# 6. Ens trobem a l'estació Marina L1 i volem arribar a l'estació Sagrada Familia L2. Si la propera estació que explorarem serà Glories L1, quin serà el seu valor d'heurística(H) respecte el criteri de TEMPS?
marinal1 = 5
gloriesl1 = 4
sfl2 = 10
temps = calculate_heuristics([Path([marinal1,gloriesl1])], map, sfl2, type_preference=1)[0].h
print("6. ", temps)