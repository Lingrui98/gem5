import pandas as pd

time_stat = 'time_stat.csv'


# Currently sort benchmarks by running time
def optimize(nun_thread, benchmarks=None, bp=None):
    b = []
    if benchmarks == None:
        with open('all_function_spec.txt') as f:
            for line in f:
                b.append(line.strip())
    else:
        b = benchmarks
    if bp == None:
        bp = 'MyPerceptron'

    df = pd.DataFrame(pd.read_csv('time_stat.csv', index_col=0))
    # Get stats of current bp
    mean = df.loc[bp].mean()

    sorted_mean = mean.sort_values(ascending=False)
    print(list(sorted_mean.index))
    return list(sorted_mean.index)

if __name__ == '__main__':
    optimize()