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
print(calculate_cost([Path([5,4,3,2,1,7,8])], map, 2)[0].g)
