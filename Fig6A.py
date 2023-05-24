import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

bar_4=[]
bar_6=[]
bar_8=[]
bar_plot=[]
significance_test=[]
p_value_list=[]
std_error_list=[]

with open ('Capacity_4%overlap', 'r') as f:
    line_4 = f.readlines()
    number_4 = [line.strip() for line in line_4]
    number_list_4 = number_4[0].split()
for i in range (len(number_list_4)):
    bar_4.append(float(number_list_4[i]))
average_4 = sum(bar_4) / len(bar_4)
significance_test.append(bar_4)
bar_plot.append(average_4)

with open ('Capacity_15HC', 'r') as f:
    line_6 = f.readlines()
    number_6 = [line.strip() for line in line_6]
    number_list_6 = number_6[0].split()
for i in range (len(number_list_6)):
    bar_6.append(float(number_list_6[i]))
average_6 = sum(bar_6) / len(bar_6)
significance_test.append(bar_6)
bar_plot.append(average_6)

with open ('Capacity_20HC.txt', 'r') as f:
    line_8 = f.readlines()
    number_8 = [line.strip() for line in line_8]
    number_list_8 = number_8[0].split()
for i in range (len(number_list_8)):
    bar_8.append(float(number_list_8[i]))
average_8 = sum(bar_8) / len(bar_8)
significance_test.append(bar_8)
bar_plot.append(average_8)

# Initialise a list of combinations of groups that are significantly different
significant_combinations = []
# Check from the outside pairs of boxes inwards
ls = list(range(1, len(significance_test) + 1))
combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
for combination in combinations:
    data1 = significance_test[combination[0] - 1]
    data2 = significance_test[combination[1] - 1]
    # Significance test (paired t-test)
    t_statistic, p_value = stats.ttest_ind(data1, data2)
    p_value_list.append(p_value)

# add stat significance in form of asterixes
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]
    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)
    y = max(ly, ry) + dh
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')
    
    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs
        
    plt.text(*mid, text, **kwargs)

# include std error
for i in range(len(bar_plot)):
    std_error = np.std(significance_test[i], ddof=1) / np.sqrt(len(significance_test[i]))
    std_error_list.append(std_error)

# customize axes
fig, ax = plt.subplots()

x_description=["no overlap"]
for i in range(1, len(bar_plot)):
    x_description.append(str(i*2) + "%")
for i in range (len(x_description)):
    ax.bar(x_description[i], bar_plot[i], width=.8, yerr=std_error_list[i], capsize=4)
ax.set_ylabel("Capacity")
ax.set_xlabel("Number of HCs")

bars = np.arange(len(bar_plot))
dh=[]
for i in range(len(combinations)):
    dh_i = .025
    how_far_apart= (combinations[i][1]-1) - (combinations[i][0]-1)
    for j in range (1, len(bar_plot)):
        if how_far_apart > j:
            dh_i += .15
    dh.append(dh_i)
    #if p_value_list[i] < .05:
        #barplot_annotate_brackets(combinations[i][0]-1, combinations[i][1]-1, p_value_list[i], bars, bar_plot, barh=.025, dh=dh[i], maxasterix=3)

barplot_annotate_brackets(combinations[0][0]-1, combinations[0][1]-1, p_value_list[0], bars, bar_plot, barh=.025, dh=.095, maxasterix=3)
barplot_annotate_brackets(combinations[1][0]-1, combinations[1][1]-1, p_value_list[1], bars, bar_plot, barh=.025, dh=.045, maxasterix=3)
barplot_annotate_brackets(combinations[2][0]-1, combinations[2][1]-1, p_value_list[2], bars, bar_plot, barh=.025, dh=.035, maxasterix=3)

plt.ylim(0,1.13)
plt.show()