import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("AB.csv")  # reads the csv file into a pandas dataframe


def pull_data(column_name):  # pulls the according column
    templist = df[column_name].astype(float)
    return np.array(templist.tolist())


# turn each column into a numpy array
Years = pull_data('Year')
Months = pull_data('Month')
Time = Years + (Months-1) / 12
MaxTemp = pull_data('MaxTemp')
MinTemp = pull_data('MinTemp')
FrostDays = pull_data('FrostDays')
Rain = pull_data('Rain')
SunHours = pull_data('SunHours')

fig, ax = plt.subplots()


def calculate_coefficients(t, data):
    #  modelling as (a + b*t + c*cos(wt) + d*sin(wt))
    #  determine x=[a,b,c,d] from Ax=b, where b is known, and A comes from the model above
    freq = 2*np.pi  # A frequency term, determines how quickly the sin and cos terms oscillate
    ones = np.ones(len(t))
    sin_time = np.sin(freq * t)
    cos_time = np.cos(freq * t)
    A1 = np.vstack([ones, t, cos_time, sin_time]).T  # defines the matrix A
    a, b, c, d = np.linalg.lstsq(A1, data, rcond=None)[0]  # Solves ||Ax-b||_2 for x
    return a, b, c, d, freq


print(calculate_coefficients(Time, SunHours))


def plot_data(t, data, name):
    a, b, c, d, w = calculate_coefficients(t, data)  # calculate the coefficients for the model

    def f(t):
        return a + b * t + c * np.cos(w * t) + d * np.sin(w * t)
    t_smooth = np.linspace(1941, 2022.5, 50000)  # smooth time
    y = f(t_smooth)
    ax.plot(t, data, label="Actual", linewidth=0.2)  # plot the actual data
    ax.plot(t_smooth, y, label="Model", linewidth=0.2)  # plot the model
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel(name)
    plt.legend()
    plt.savefig(f"{name}_graph.pdf")
    plt.clf()


plot_data(Time, SunHours, "SunHours")