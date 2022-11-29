import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

class apply_pyplot:
    def run(x, y, commit, alg_list):
        fig, ax=plt.subplots(nrows=1, ncols=1)

        for i in range(0, 6):
            ax.bar(x, y[i, :], label=alg_list[i])

        ax.set(title='Prediction('+commit+')',
        xlabel='Player Lists',
        ylabel='Expected Annual Income')
        ax.set_xticks(x)
        fig.set_size_inches(15, 7)

        plt.legend(
            shadow=True,
            fancybox=False,
            loc="upper right",
            framealpha=0.1
        )
        plt.savefig('.\\Data_visualization_\\result_visual\\'+commit+'_total_bar.jpg')
        plt.show()

        print('잠시만 기다려 주십시오. Please wait...')
        
        for i in range(0, 6):
            fig, ax=plt.subplots(nrows=1, ncols=1)
            ax.bar(x, y[i, :], label=alg_list[i])
            ax.set(title='Prediction('+commit+')',
            xlabel='Player Lists',
            ylabel='Expected Annual Income')
            ax.set_xticks(x)
            fig.set_size_inches(15, 7)
            plt.savefig('.\\Data_visualization_\\result_visual\\'+commit+'_'+alg_list[i]+'_bar.jpg')
            plt.close()