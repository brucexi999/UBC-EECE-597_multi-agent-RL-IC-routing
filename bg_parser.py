def bg_parser():
    filepath = "/home/brucexi2/DQN_GlobalRouting/GlobalRoutingRL/benchmark/test_benchmark_1.gr"
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


bg_parser()
