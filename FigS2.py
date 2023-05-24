import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

bar_0=[]
bar_1=[]
bar_2=[]
bar_3=[]
bar_4=[]
bar_5=[]
bar_6=[]
bar_7=[]
bar_8=[]
bar_9=[]
bar_10=[]
bar_plot=[]
significance_test=[]
p_value_list=[]
std_error_list=[]

# import obtained values from the simulations
with open ('Synaptic_weights_wo0_0%overlap_dis_allMC', 'r') as f:
    line_0 = f.readlines()
    number_0 = [line.strip() for line in line_0]
    number_list_0 = number_0[0].split()
for i in range (len(number_list_0)):
    bar_0.append(float(number_list_0[i]))
average_0 = sum(bar_0) / len(bar_0)
significance_test.append(bar_0)
bar_plot.append(average_0)

with open ('Synaptic_weights_wo0_2%overlap_dis_allMC', 'r') as f:
    line_2 = f.readlines()
    number_2 = [line.strip() for line in line_2]
    number_list_2 = number_2[0].split()
for i in range (len(number_list_2)):
    bar_2.append(float(number_list_2[i]))
average_2 = sum(bar_2) / len(bar_2)
significance_test.append(bar_2)
bar_plot.append(average_2)

with open ('Synaptic_weights_wo0_4%overlap_dis_allMC', 'r') as f:
    line_4 = f.readlines()
    number_4 = [line.strip() for line in line_4]
    number_list_4 = number_4[0].split()
for i in range (len(number_list_4)):
    bar_4.append(float(number_list_4[i]))
average_4 = sum(bar_4) / len(bar_4)
significance_test.append(bar_4)
bar_plot.append(average_4)

with open ('Synaptic_weights_wo0_6%overlap_dis_allMC', 'r') as f:
    line_6 = f.readlines()
    number_6 = [line.strip() for line in line_6]
    number_list_6 = number_6[0].split()
for i in range (len(number_list_6)):
    bar_6.append(float(number_list_6[i]))
average_6 = sum(bar_6) / len(bar_6)
significance_test.append(bar_6)
bar_plot.append(average_6)

with open ('Synaptic_weights_wo0_8%overlap_dis_allMC.txt', 'r') as f:
    line_8 = f.readlines()
    number_8 = [line.strip() for line in line_8]
    number_list_8 = number_8[0].split()
for i in range (len(number_list_8)):
    bar_8.append(float(number_list_8[i]))
average_8 = sum(bar_8) / len(bar_8)
significance_test.append(bar_8)
bar_plot.append(average_8)

with open ('Synaptic_weights_wo0_10%overlap_dis_allMC', 'r') as f:
    line_10 = f.readlines()
    number_10 = [line.strip() for line in line_10]
    number_list_10 = number_10[0].split()
for i in range (len(number_list_10)):
    bar_10.append(float(number_list_10[i]))
average_10 = sum(bar_10) / len(bar_10)
significance_test.append(bar_10)
bar_plot.append(average_10)

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
ax.set_xlabel("overlap percentage")

plt.ylim(0,4.5)
plt.show()