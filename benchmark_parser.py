import os
import numpy as np


def bg_parser(filepath):
    with open(filepath) as file:
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
    length, width = int(lines[0].split()[1]), int(lines[0].split()[2])
    capacity = int(lines[1].split()[-1])
    n_nets = int(lines[7].split()[-1])
    netlist = [[]for _ in range(n_nets)]
    net_counter = 0
    net_line_index = 8
    while net_counter < n_nets:
        n_pins = int(lines[net_line_index].split()[-2])
        for i in range(net_line_index+1, net_line_index+1+n_pins):
            x, y = int(int(lines[i].split()[0]) / 10), int(int(lines[i].split()[1]) / 10)
            netlist[net_counter].append((x, y))
        net_counter += 1
        net_line_index += (n_pins+1)

    return length, width, capacity, netlist


def parser(filepath):
    with open(filepath) as file:
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
    length, width = lines[0].split()
    length, width = int(length), int(width)
    n_macros = int(lines[1])
    macros = []
    for i in range(2, 2+n_macros):
        macro_x, macro_y = lines[i].split()
        macros.append((int(macro_x), int(macro_y)))
    n_nets_index = i+1
    n_nets = int(lines[n_nets_index])
    first_net_index = n_nets_index+1
    nets = []
    for i in range(first_net_index, first_net_index+n_nets):
        net = []
        split_net_line = lines[i].split()
        n_pins = int(split_net_line[0])
        j = 1
        while (j < n_pins*2):
            pin_x, pin_y = int(split_net_line[j]), int(split_net_line[j+1])
            net.append((pin_x, pin_y))
            j += 2
        nets.append(net)

    return length, width, macros, nets


def all_parser(directory_path, congestion=True):
    benchmarks = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    env_config = []
    for benchmark in benchmarks:
        length, width, macros, nets = parser(directory_path+"/{}".format(benchmark))
        n_nets = len(nets)
        edge_capacity = np.full((length, width, 4), n_nets - int(congestion))
        config_dict = {
            "length": length,
            "width": width,
            "nets": nets,
            "macros": macros,
            "edge_capacity": edge_capacity
        }
        env_config.append(config_dict)

    return env_config
