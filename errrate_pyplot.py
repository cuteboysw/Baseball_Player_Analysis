import matplotlib.pyplot as plt
import numpy as np

class errrate_pyplot:
    def run(x, y, ycnt):
        fig, ax = plt.subplots(nrows=2, ncols=4)

        ax[0, 0].plot(x, y[:, 0])
        ax[0, 0].set_title('SGD')

        ax[0, 1].plot(x, y[:, 1])
        ax[0, 1].set_title('Linear\nRegression')

        ax[0, 2].plot(x, y[:, 2])
        ax[0, 2].set_title('Lasso')

        ax[0, 3].plot(x, y[:, 3])
        ax[0, 3].set_title('Ridge')

        ax[1, 0].plot(x, y[:, 4])
        ax[1, 0].set_title('Logistic\nRegression')

        ax[1, 1].plot(x, y[:, 5])
        ax[1, 1].set_title('K-Neighbors\nRegression')

        ax[1, 2].plot(x, y[:, 6])
        ax[1, 2].set_title('Decision Tree')

        plt.subplots_adjust(hspace=0.6, wspace=0.5)

        plt.savefig('.\\error_rate\\error_rate_pyplot.jpg')
        plt.show()
