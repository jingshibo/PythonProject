## import
import matplotlib.pyplot as plt
import pandas as pd


##  plot the bars with all columns compared to the first column
def plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1):
    # Create sample data
    df_mean = reorganized_results[dataset]['mean']
    df_std = reorganized_results[dataset]['std']
    df_pval = reorganized_results[dataset]['ttest']
    df_mean.columns = legend
    df_std.columns = legend
    df_pval.columns = legend
    font_size = 20

    # Create color list
    color_list = ['steelblue', 'wheat', 'darkorange', 'yellowgreen', 'pink', 'darkgray', 'lawngreen', 'cornflowerblue', 'gold', 'slategray']
    # Plot bar chart with error bars
    ax = df_mean.plot.bar(yerr=df_std, capsize=4, width=0.8, color=color_list)
    # Correcting p_val based on the number of group pairs to compare
    bonferroni_correction = bonferroni_coeff  # it is too conservative

    # assume all bars have the same width
    bar_width = ax.patches[0].get_width()
    # iterate over the columns and rows of df_pval to plot p_val
    for j in range(df_pval.shape[1]):
        for i in range(df_pval.shape[0]):
            pval = df_pval.iloc[i, j]
            if not pd.isna(pval):
                # Note: ploting of bars are following the order of looping through all the rows of the first column and then the second columns
                # ac.patches include the position of each bars, but the order is ploting the first column at each x-label, and then the second column at each x-label.
                # this can be verified by printing: ax.patches[0].get_x(), ax.patches[1].get_x(), ax.patches[2].get_x().. to check the value of x position
                # for each x-label, the position is reset rather than inherit.
                # calculate the x and y coordinates of the horizontal lines
                x_left, y_left = ax.patches[i].get_x() + bar_width / 2, df_mean.iloc[i, 0]
                x_right, y_right = ax.patches[j * df_pval.shape[0] + i].get_x() + bar_width / 2, df_mean.iloc[i, j]

                # calculate the height and y coordinate of the significance stars
                height = max(df_mean.iloc[i, :])
                line_height = height + df_std.iloc[i, 0] + 2 * j
                y_left, y_right = line_height, line_height
                # add the line ot the plot
                ax.plot([x_left, x_right], [y_left, y_right], 'k-', lw=1)

                # add the significance star ot the plot
                if pval < 0.01 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.8, "**", ha='center', va='bottom', color='r', fontsize=20)
                elif pval < 0.05 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.8, "*", ha='center', va='bottom', color='r', fontsize=20)
                # elif pval < 0.1 / bonferroni_correction:
                #     ax.text((x_left + x_right) * 0.5, star_height, "*", ha='center', va='bottom', color='r', fontsize=15)
                else:  # if not significant
                    ax.text((x_left + x_right) * 0.5, line_height, "ns", ha='center', va='bottom', color='r', fontsize=15)

                # add small vertical bars at two ends of the lines
                left_line = ax.plot([x_left, x_left], [y_left - 0.005 * height, y_left], 'k-', lw=1)
                right_line = ax.plot([x_right, x_right], [y_right - 0.005 * height, y_right], 'k-', lw=1)

    # Set x-axis
    x_label = ax.set_xlabel('Prediction time delay(ms)', fontsize=font_size)  # Set x-axis label
    # Set the x-tick positions and labels
    # x_ticks = list(range(len(ax.get_xticklabels())))
    # ax.set_xticks(x_ticks)
    # set x-tick values
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_ms = [string[string.find('_')+1: string.find('_', string.find('_')+1)] for string in x_tick_labels]  # extract only the delay value
    ax.set_xticklabels(x_tick_ms, rotation=0)

    # Set y-axis limit
    if df_mean.shape[1] == 5:
        ax.set_ylim(35, 110)  # for losing channels
    elif df_mean.shape[1] == 6:
        ax.set_ylim(50, 115)  # for one muscle channels
    elif df_mean.shape[1] == 10:
        ax.set_ylim(50, 120)  # for reduced channels
    ymin, ymax = ax.get_ylim()  # get the current limits of the y-axis
    yticks = range(int(ymin), int(ymax+1), 5)  # set the space between y-axis ticks to 5
    ax.set_yticks(yticks)
    ax.set_ylabel('Prediction accuracy(%)', fontsize=font_size)  # Set y-axis label

    # Set figure display
    ax.tick_params(axis='both', labelsize=font_size)
    ax.set_title(title)  # set figure title
    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=font_size)  # Set legend position
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=font_size-2)
    fig = plt.gcf()  # gets the current figure instance and assigns it to the variable fig
    fig.set_size_inches(8, 6)  # sets the size of the figure to be 6 inches by 6 inches
    fig.subplots_adjust(right=0.8)  # adjusts the spacing between the subplots in the figure
    # Show plot
    plt.show()


##  plot the bars with all columns compared to the previous column
def plotAdjacentTtest(reorganized_results, dataset, legend, title, bonferroni_coeff=1):
    # Create sample data
    df_mean = reorganized_results[dataset]['mean']
    df_std = reorganized_results[dataset]['std']
    df_pval = reorganized_results[dataset]['ttest']
    df_mean.columns = legend
    df_std.columns = legend
    df_pval.columns = legend
    font_size = 20

    # Create color list
    color_list = ['steelblue', 'wheat', 'darkorange', 'yellowgreen', 'pink', 'darkgray', 'lawngreen', 'cornflowerblue', 'gold', 'slategray']
    # Plot bar chart with error bars
    ax = df_mean.plot.bar(yerr=df_std, capsize=4, width=0.8, color=color_list)
    # Correcting p_val based on the number of group pairs to compare
    bonferroni_correction = bonferroni_coeff

    # assume all bars have the same width
    bar_width = ax.patches[0].get_width()
    # iterate over the columns and rows of df_pval to plot p_val
    for j in range(df_pval.shape[1]):
        for i in range(df_pval.shape[0]):
            pval = df_pval.iloc[i, j]
            if not pd.isna(pval):
                # calculate the x and y coordinates of the horizontal lines
                x_left, y_left = ax.patches[(j-1) * df_pval.shape[0] + i].get_x() + bar_width / 2, df_mean.iloc[i, j-1]
                x_right, y_right = ax.patches[j * df_pval.shape[0] + i].get_x() + bar_width / 2, df_mean.iloc[i, j]

                # calculate the height and y coordinate of the significance stars
                height = max(df_mean.iloc[i, :])
                line_height = height + df_std.iloc[i, 0] + 2 * j
                y_left, y_right = line_height, line_height
                # add the line ot the plot
                ax.plot([x_left, x_right], [y_left, y_right], 'k-', lw=1)

                # add the line and significance stars ot the plot
                ax.plot([x_left, x_right], [y_left, y_right], 'k-', lw=1)
                if pval < 0.01 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "**", ha='center', va='bottom', color='r', fontsize=25)
                elif pval < 0.05 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "*", ha='center', va='bottom', color='r', fontsize=25)
                # elif pval < 0.1 / bonferroni_correction:
                #     ax.text((x_left + x_right) * 0.5, star_height, "*", ha='center', va='bottom', color='r', fontsize=15)
                else:  # if not significant
                    ax.text((x_left + x_right) * 0.5, line_height, "ns", ha='center', va='bottom', color='r', fontsize=20)

                # add small vertical bars at two ends of the lines
                left_line = ax.plot([x_left, x_left], [y_left - 0.005 * height, y_left], 'k-', lw=1)
                right_line = ax.plot([x_right, x_right], [y_right - 0.005 * height, y_right], 'k-', lw=1)

    # Set x-axis
    x_label = ax.set_xlabel('Prediction time delay(ms)', fontsize=font_size)  # Set x-axis label
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_ms = [string[string.find('_')+1: string.find('_', string.find('_')+1)] for string in x_tick_labels]  # extract only the delay value
    ax.set_xticklabels(x_tick_ms, rotation=0)  # set x-tick value

    # Set y-axis
    ax.set_ylim(60, None)  # only set the lower limit
    ymin, ymax = ax.get_ylim()  # get the current limits of the y-axis
    yticks = range(int(ymin), int(ymax+1), 5)  # set the space between y-axis ticks to 5
    ax.set_yticks(yticks)
    ax.set_ylabel('Prediction accuracy(%)', fontsize=font_size)  # Set y-axis label

    # set figure display
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=font_size)
    ax.set_title(title)  # Set plot title
    # Show plot
    plt.show()

