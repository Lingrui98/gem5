import pandas as pd
import os
import random

time_stat = 'time_stat.csv'

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
    print(bp)
    try:
        df = pd.DataFrame(pd.read_csv('time_stat.csv', index_col=0))
    except:
        print("No time stat file, returning input")
        return b
    # Select given benchmarks
    df = df[b]
    b_copy = b.copy()

    try:
        # Get stats of current bp
        stat = df.loc[bp]
        if type(stat) == type(df):
            stat = stat.mean()
        before = estimated_t(stat, num_thread)
        print('Given order has estimated running time %.2fmin' % (before / 60.0))
        
        # Sort by running time
        sorted_mean = stat.sort_values(ascending=False)
        optimized_b = list(sorted_mean.index)
        after = estimated_t(sorted_mean, num_thread)
        print('Descending order gives estimated running time %.2fmin' % (after / 60.0))
        
        # Random order
        itr = 1000
        # print(df)
        fastest = -1
        order = []
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
        
        optimal = estimated_t(stat, 1) / float(num_thread)
        print('Optimal running time is %.2fmin' % (optimal / 60.0))
        
        if fastest < before and fastest < after:
            print("Using random optimized order")
            return order
        elif after < before:
            print("Using optimized order")
            return optimized_b
        else:
            print("Using given order")
            return b
    except:
        # No data of current bp, return with input order
        print('No data of current bp.')
        return b


if __name__ == '__main__':
    optimize(5)