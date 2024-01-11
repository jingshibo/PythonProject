import copy
import matplotlib.pyplot as plt
import pandas as pd


##  plot the bars with all columns compared to the previous column for the significance level annotation
def plotSubjectAdjacentTtest(mean_std_value, legend, columns_to_plot, title, bonferroni_coeff=1):
    # Create sample data
    data = copy.deepcopy(mean_std_value)
    df_mean = data['accuracy']['statistics']['cm_diagonal_mean'][columns_to_plot]
    df_std = data['accuracy']['statistics']['std'][columns_to_plot]
    df_pval = data['accuracy']['statistics']['ttest'][columns_to_plot]
    df_mean.columns = legend
    df_std.columns = legend
    df_pval.columns = legend
    font_size = 30

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
            if not pd.isna(pval) and not (i == 0 and (j == 0)):
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
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "**", ha='center', va='bottom', color='r', fontsize=35)
                elif pval < 0.05 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "*", ha='center', va='bottom', color='r', fontsize=35)
                # elif pval < 0.1 / bonferroni_correction:
                #     ax.text((x_left + x_right) * 0.5, star_height, "*", ha='center', va='bottom', color='r', fontsize=15)
                else:  # if not significant
                    ax.text((x_left + x_right) * 0.5, line_height, "ns", ha='center', va='bottom', color='r', fontsize=30)

                # add small vertical bars at two ends of the lines
                left_line = ax.plot([x_left, x_left], [y_left - 0.005 * height, y_left], 'k-', lw=1)
                right_line = ax.plot([x_right, x_right], [y_right - 0.005 * height, y_right], 'k-', lw=1)

    # Set x-axis
    x_label = ax.set_xlabel('Model Training Methods', fontsize=font_size)  # Set x-axis label
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_ms = [string[string.find('_')+1: string.find('_', string.find('_')+1)] for string in x_tick_labels]  # extract only the delay value
    ax.set_xticklabels([])  # set x-tick value

    # Set y-axis
    ax.set_ylim(70, None)  # only set the lower limit
    ymin, ymax = ax.get_ylim()  # get the current limits of the y-axis
    yticks = range(int(ymin), int(ymax+1), 5)  # set the space between y-axis ticks to 5
    ax.set_yticks(yticks)
    # Adjust the y-tick labels to hide values greater than 100
    new_ytick_labels = [label if label <= 100 else '' for label in ax.get_yticks()]
    ax.set_yticklabels(new_ytick_labels)
    ax.set_ylabel('Prediction Accuracy(%)', fontsize=font_size)  # Set y-axis label

    # set figure display
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(loc='upper right', fontsize=font_size-4)
    ax.set_title(title, fontsize=font_size)  # Set plot title
    # Show plot
    plt.show()


##  plot the bars with all columns compared to the previous column for the significance level annotation
def plotNumOfReferenceAdjacentTtest(mean_std_value, benchmark_mean_std_value, legend, columns_to_plot, title, bonferroni_coeff=1):
    # Create sample data
    data = copy.deepcopy(mean_std_value)
    df_mean = data['accuracy']['statistics']['cm_diagonal_mean'][columns_to_plot]
    df_std = data['accuracy']['statistics']['std'][columns_to_plot]
    df_pval = data['accuracy']['statistics']['ttest'][columns_to_plot]
    df_mean.columns = legend
    df_std.columns = legend
    df_pval.columns = legend
    font_size = 35

    # bench mark values
    benchmark = copy.deepcopy(benchmark_mean_std_value)
    lowest_benchmark = benchmark['accuracy']['statistics']['cm_diagonal_mean']['accuracy_worst'].to_numpy()
    tf_benchmark = benchmark['accuracy']['statistics']['cm_diagonal_mean']['accuracy_tf'].to_numpy()
    highest_benchmark = benchmark['accuracy']['statistics']['cm_diagonal_mean']['accuracy_best'].to_numpy()

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
            if not pd.isna(pval) and not (i == 0 and (j == 3 or j == 4)):
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
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "**", ha='center', va='bottom', color='r', fontsize=35)
                elif pval < 0.05 / bonferroni_correction:
                    ax.text((x_left + x_right) * 0.5, line_height - 0.5, "*", ha='center', va='bottom', color='r', fontsize=35)
                # elif pval < 0.1 / bonferroni_correction:
                #     ax.text((x_left + x_right) * 0.5, star_height, "*", ha='center', va='bottom', color='r', fontsize=15)
                else:  # if not significant
                    ax.text((x_left + x_right) * 0.5, line_height, "ns", ha='center', va='bottom', color='r', fontsize=30)

                # add small vertical bars at two ends of the lines
                left_line = ax.plot([x_left, x_left], [y_left - 0.005 * height, y_left], 'k-', lw=1)
                right_line = ax.plot([x_right, x_right], [y_right - 0.005 * height, y_right], 'k-', lw=1)

    # Plot horizontal lines for benchmarks
    ax.axhline(y=tf_benchmark, color='green', linestyle='-.')
    ax.axhline(y=lowest_benchmark, color='blue', linestyle='--')
    # ax.axhline(y=highest_benchmark, color='blue', linestyle='-', label='Highest Benchmark')

    # Set x-axis
    x_label = ax.set_xlabel('Number of New Data Available Per Transition Mode', fontsize=font_size)  # Set x-axis label
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    x_tick_ms = [f"{label.split('_')[-1]}" for label in x_tick_labels]  # extract only the delay value
    ax.set_xticklabels(x_tick_ms, rotation=0)  # set x-tick value

    # Set y-axis
    ax.set_ylim(70, None)  # only set the lower limit
    ymin, ymax = ax.get_ylim()  # get the current limits of the y-axis
    yticks = range(int(ymin), int(ymax+1), 5)  # set the space between y-axis ticks to 5
    ax.set_yticks(yticks)
    # Adjust the y-tick labels to hide values greater than 100
    new_ytick_labels = [label if label <= 100 else '' for label in ax.get_yticks()]
    ax.set_yticklabels(new_ytick_labels)
    ax.set_ylabel('Classification Accuracy(%)', fontsize=font_size)  # Set y-axis label

    # set figure display
    ax.tick_params(axis='both', labelsize=font_size)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=font_size - 4, ncol=7)
    ax.set_title(title, fontsize=font_size)  # Set plot title
    # Show plot
    plt.show()


##  plot the bars with all columns compared to the previous column for the significance level annotation
def plotModeAccuracyAdjacentTtest(mean_std_value, legend, columns_to_plot, title, bonferroni_coeff=1):
    # Create sample data
    data = copy.deepcopy(mean_std_value)
    df_mean = data['accuracy']['statistics']['mean'][columns_to_plot]
    df_mean.columns = legend
    font_size = 35

    # Create color list
    color_list = ['steelblue', 'wheat', 'darkorange', 'yellowgreen', 'pink', 'darkgray', 'lawngreen', 'cornflowerblue', 'gold',
        'slategray']
    ax = df_mean.plot.bar(capsize=4, width=0.8, color=color_list)

    # Customizations
    plt.title(title, fontsize=font_size+2)
    plt.xlabel('Transition Mode', fontsize=font_size)
    plt.ylabel('Classification Accuracy(%)', fontsize=font_size)
    plt.xticks(range(len(df_mean.index)), df_mean.index, rotation=0, fontsize=font_size)  # Set x-tick labels as row index names
    plt.yticks(fontsize=font_size)
    ax.set_ylim(55, 100)
    # Display the plot

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), fontsize=font_size - 5, ncol=3)  # Adjust legend
    # plt.tight_layout()  # Adjust layout for saving or displaying
    plt.show()
