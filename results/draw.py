import csv

import pandas as pd
import matplotlib.pyplot as plt


def show(data, title):
    df = pd.DataFrame(data)
    df.plot.box(title="Consumer spending in each country")
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()


def calc_qerror(file_name):
    qerror = []
    with open(file_name, 'r') as f:
        rows = [list(v) for v in csv.reader(f)]
        for row in rows:
            predict, real = row



def draw_cost_fed():
    data = {}


if __name__ == '__main__':
    draw_cost_fed()
