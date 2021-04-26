performance_by_seed = []

while True:
    try:
        x = input()
    except EOFError:
        exit()
    split_inp = x.split(',')
    performance_by_seed.append([split_inp[3], float(split_inp[5]), x]) 
    if split_inp[3] == '10':
        sorted_perf = sorted(performance_by_seed, key=(lambda item: item[1]))
        top_seeds = [top_perf[0] for top_perf in sorted_perf[-3:]]
        print(','.join(split_inp[:3]) + ',' + str(top_seeds))
        performance_by_seed = []
