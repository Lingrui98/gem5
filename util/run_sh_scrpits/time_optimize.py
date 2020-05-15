import pandas as pd
import os
import random
import traceback

time_stat = 'time_stat.csv'

# Sort by running time in descending order
def sort_descending(stat, num_thread):
    sort = stat.sort_values(ascending=False)
    sorted_b = list(sort.index)
    after = estimated_t(sort, num_thread)
    print('Descending order gives estimated running time %.2fmin' % (after / 60.0))
    return sorted_b, after

# Randomly reorder benchmarks and choose the best order
def random_optimizing(itr, df, b_init, bp, num_thread):
    fastest = -1
    order = []
    b = b_init
    for i in range(itr):
        random.shuffle(b)
        df = df[b]
        stat = df.loc[bp]
        if type(stat) == type(df):
            stat = stat.mean()
        before = estimated_t(stat, num_thread)
        if fastest == -1 or before < fastest:
            fastest = before
            order = b
    print('Random optimization gives running time of %.2fmin' % (fastest / 60.0))
    return order, fastest

def estimated_t(sorted_mean, n):
    t_list = [0 for i in range(n)]
    ptr = 0
    total = 0
    # Simulate the simulation process
    for t in sorted_mean:
        # print(t_list)
        fastest = -1
        flag = False
        ptr = -1
        # Find the first thread that has done its work
        for i in range(n):
            # Record the thread with the least remaining time
            if fastest == -1:
                fastest = t_list[i]
            else:
                if t_list[i] < fastest:
                    fastest = t_list[i]
                    ptr = i
            # If some thread has done its work, give it this job
            # and switch to the next job
            if t_list[i] == 0:
                flag = True
                t_list[i] += t
                break
        if flag:
            continue
        else:
            # print("fastest is", fastest)
            for i in range(len(t_list)):
                t_list[i] -= fastest
                assert(t_list[i] >= 0)
            total += fastest
            t_list[ptr] += t
    total += max(t_list)
    return total

# Currently sort benchmarks by running time
def optimize(num_thread, benchmarks=None, bp=None):
    b = []
    if benchmarks == None:
        with open('all_function_spec.txt') as f:
            for line in f:
                b.append(line.strip())
    else:
        b = benchmarks
    if bp == None:
        bp = 'MyPerceptron'

    try:
        df = pd.DataFrame(pd.read_csv('time_stat.csv', index_col=0))
    except:
        print("No time stat file, returning input")
        return b
    # Select given benchmarks
    df = df[b]
    b_copy = b.copy()

    best = b
    try:
        # Get stats of current bp
        stat = df.loc[bp]
        if type(stat) == type(df):
            stat = stat.mean()

        before = estimated_t(stat, num_thread)
        print('Given order has estimated running time %.2fmin' % (before / 60.0))
        
        # Sort by running time
        sorted_b, t_sort = sort_descending(stat, num_thread)

        # Random order
        itr = 1000
        random_b, t_rand = random_optimizing(itr, df, b, bp, num_thread)

        # Calculate lower bound
        lb = estimated_t(stat, 1) / float(num_thread)
        # Amdahl, the slowest work
        lb = stat.max() if stat.max() > lb else lb
        print('Lower bound running time is %.2fmin' % (lb / 60.0))
        
        # Choose the best order
        if t_rand < before and t_rand < t_sort:
            print("Using random optimized order\n")
            best = random_b
        elif t_sort < before:
            print("Using optimized order\n")
            best = sorted_b
        else:
            print("Using given order\n")
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        return best


if __name__ == '__main__':
    for i in range(8):
        print("For", i+1, "threads,")
        optimize(i+1, bp='LTAGE')