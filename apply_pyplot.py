import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class apply_pyplot:
    def run(x, y, commit, alg_list):
        fig, ax=plt.subplots(nrows=1, ncols=1)

        for i in range(0, 6):
            ax.plot(x, y[i, :], label=alg_list[i])

        ax.set(title='Prediction('+commit+')',
        xlabel='Player Lists',
        ylabel='Expected Annual Income')
        ax.set_xticks(x)
        fig.set_size_inches(15, 7)
        
        plt.legend(
            shadow=True,
            fancybox=False,
            loc="upper right"
        )
        plt.savefig('.\\result\\'+commit+'_pyplot.jpg')
        plt.show()
        