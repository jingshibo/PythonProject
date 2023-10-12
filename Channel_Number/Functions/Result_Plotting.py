## import
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy


##  plot the bars with all columns compared to the first column for the significance level annotation
def plotCompareToFirstTtest(reorganized_results, dataset, legend, title='', bonferroni_coeff=1):
    # Create sample data
    data = copy.deepcopy(reorganized_results)
    df_mean = data[dataset]['mean']
    df_std = data[dataset]['std']
    df_pval = data[dataset]['ttest']
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
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "**", ha='center', va='bottom', color='r', fontsize=20)
                elif pval < 0.05 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "*", ha='center', va='bottom', color='r', fontsize=20)
                # elif pval < 0.1 / bonferroni_correction:
                #     ax.text((x_left + x_right) * 0.5, star_height, "*", ha='center', va='bottom', color='r', fontsize=15)
                else:  # if not significant
                    ax.text((x_left + x_right) * 0.5, line_height, "ns", ha='center', va='bottom', color='r', fontsize=15)

                # add small vertical bars at two ends of the lines
                left_line = ax.plot([x_left, x_left], [y_left - 0.005 * height, y_left], 'k-', lw=1)
                right_line = ax.plot([x_right, x_right], [y_right - 0.005 * height, y_right], 'k-', lw=1)

    # Set x-axis
    x_label = ax.set_xlabel('Prediction Delay Relative to The Toe-off Moment(ms)', fontsize=font_size)  # Set x-axis label
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
    # Adjust the y-tick labels to hide values greater than 100
    new_ytick_labels = [label if label <= 100 else '' for label in ax.get_yticks()]
    ax.set_yticklabels(new_ytick_labels)
    ax.set_ylabel('Prediction Accuracy(%)', fontsize=font_size)  # Set y-axis label

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


##  plot the bars with all columns compared to the previous column for the significance level annotation
def plotAdjacentTtest(reorganized_results, dataset, legend, title, bonferroni_coeff=1):
    # Create sample data
    data = copy.deepcopy(reorganized_results)
    df_mean = data[dataset]['mean']
    df_std = data[dataset]['std']
    df_pval = data[dataset]['ttest']
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
    x_label = ax.set_xlabel('Prediction Delay Relative to The Toe-off Moment(ms)', fontsize=font_size)  # Set x-axis label
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_ms = [string[string.find('_')+1: string.find('_', string.find('_')+1)] for string in x_tick_labels]  # extract only the delay value
    ax.set_xticklabels(x_tick_ms, rotation=0)  # set x-tick value

    # Set y-axis
    ax.set_ylim(60, None)  # only set the lower limit
    ymin, ymax = ax.get_ylim()  # get the current limits of the y-axis
    yticks = range(int(ymin), int(ymax+1), 5)  # set the space between y-axis ticks to 5
    ax.set_yticks(yticks)
    # Adjust the y-tick labels to hide values greater than 100
    new_ytick_labels = [label if label <= 100 else '' for label in ax.get_yticks()]
    ax.set_yticklabels(new_ytick_labels)
    ax.set_ylabel('Prediction Accuracy(%)', fontsize=font_size)  # Set y-axis label

    # set figure display
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=font_size)
    ax.set_title(title)  # Set plot title
    # Show plot
    plt.show()


## plot the accuracy of channel loss before and after recovery via stacked bar visualization
def plotChannelLoss(reorganized_results, dataset_1, dataset_2, legend_1, legend_2, title='', bonferroni_coeff=1):
    font_size = 30
    data = copy.deepcopy(reorganized_results)

    # Extract data for both datasets
    df_mean_1 = data[dataset_1]['mean']
    df_mean_1.columns = legend_1
    df_mean_2 = data[dataset_2]['mean']
    df_mean_2.columns = legend_1
    enhancement = df_mean_2 - df_mean_1
    enhancement.columns = legend_2
    df_mean_2.columns = legend_2
    df_std_1 = data[dataset_1]['std']
    df_std_2 = data[dataset_2]['std']
    df_pval = data[dataset_2]['ttest']
    df_pval.columns = legend_2

    # Create color list for both datasets
    color_list_1 = ['darkgray', 'wheat', 'darkorange', 'yellowgreen', 'pink']
    color_list_2 = ['slategray', 'lawngreen', 'cornflowerblue', 'gold', 'steelblue']

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create stacked bars
    n_groups = df_mean_1.shape[0]
    bar_width = 0.1
    group_spacing = 0.2 + bar_width * (df_mean_1.shape[1] - 1)  # spacing between groups of bars for each x-tick
    index = np.arange(0, n_groups * group_spacing, group_spacing)

    # plot each bar
    for col in range(df_mean_1.shape[1]):
        bar_position = index + col * bar_width
        ax.bar(bar_position, df_mean_1.iloc[:, col], bar_width, label=legend_1[col], color=color_list_1[col], yerr=df_std_1.iloc[:, col],
            capsize=5)
        if not np.all(enhancement.iloc[:, col] == 0):  # Only plot if not all values in enhancement are zero
            ax.bar(bar_position, enhancement.iloc[:, col], bar_width, bottom=df_mean_1.iloc[:, col], label=legend_2[col],
                color=color_list_2[col], yerr=df_std_2.iloc[:, col], capsize=5)


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
                x_left, y_left = ax.patches[i].get_x() + bar_width / 2, df_mean_2.iloc[i, 0]
                x_right, y_right = ax.patches[j * 2 * df_pval.shape[0] + i].get_x() + bar_width / 2, df_mean_2.iloc[i, j]

                # calculate the height and y coordinate of the significance stars
                height = max(df_mean_2.iloc[i, :])
                line_height = height + df_std_2.iloc[i, 0] + 2 * j
                y_left, y_right = line_height, line_height
                # add the line ot the plot
                ax.plot([x_left, x_right], [y_left, y_right], 'k-', lw=1)

                # add the significance star ot the plot
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
    ax.set_xlabel('Prediction Delay Relative to The Toe-off Moment(ms)', fontsize=font_size)
    ax.set_xticks(index + bar_width * (df_mean_1.shape[1] - 1) / 2)  # center the x-ticks among the bars
    ax.set_xticklabels(df_mean_1.index)
    # Adjust the x-tick labels
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_ms = [string[string.find('_')+1: string.find('_', string.find('_')+1)] for string in x_tick_labels]  # extract only the delay value
    ax.set_xticklabels(x_tick_ms, rotation=0)  # set x-tick value

    # Set y-axis
    ax.set_ylim(40, 110)
    ymin, ymax = ax.get_ylim()
    yticks = range(int(ymin), int(ymax + 1), 5)
    ax.set_yticks(yticks)
    # Adjust the y-tick labels to hide values greater than 100
    new_ytick_labels = [label if label <= 100 else '' for label in ax.get_yticks()]
    ax.set_yticklabels(new_ytick_labels)
    ax.set_ylabel('Prediction Accuracy(%)', fontsize=font_size)

    # set figure display
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=font_size - 5)
    ax.set_title(title)

    # reduce the size of the main plot to make room for the legend at the top
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])  # shrink height by 5% to make space for the legend

    plt.show()


