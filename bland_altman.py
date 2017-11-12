"""
Agreement Between Methods of Measurement with Multiple Observations Per Individual

METHOD WHERE THE TRUE VALUE VARIES

Calculations for the “non-constant” situation are relatively straightforward, using the difference between methods
for each pair. We want to estimate the mean difference and the standard deviation of differences about the mean.
To do this, we must estimate two different variances: that for repeated differences between the two methods on
the same subject and that for the differences between the averages of the two methods across subjects.
The model is that the observed difference is the sum of the mean difference (bias),
a random between subjects effect (heterogeneity) and a random error within the subject.

Reference:
Bland, J.M. and Altman, D.G., 2007. Agreement between methods of measurement with multiple observations per individual.
Journal of biopharmaceutical statistics, 17(4), pp.571-582.
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class BlandAltmanAnalysis:

    """
    Get the minimum number of observations per subject.
    Use this if the analysis requires to have the same number of observations per subject.
    """
    @staticmethod
    def _get_min_regularised_data_per_subject(data):

        min_count = min(data.groupby(['subject'])['evaluation_A'].count())
        return data.groupby('subject').head(min_count)

    @staticmethod
    def _get_antilog(log_val):
        return round(math.pow(10, log_val), 2)

    @staticmethod
    def analyse(dataframe, log_transformed=True, min_count_regularise=False):

        if min_count_regularise:
            dataframe = BlandAltmanAnalysis._get_min_regularised_data_per_subject(dataframe)

        if log_transformed:
            dataframe = dataframe.assign(evaluation_A_log_transformed=np.log10(dataframe['evaluation_A']))
            dataframe = dataframe.assign(evaluation_B_log_transformed=np.log10(dataframe['evaluation_B']))
            dataframe = dataframe.assign(diff=dataframe['evaluation_A_log_transformed'] - dataframe['evaluation_B_log_transformed'])
        else:
            dataframe = dataframe.assign(diff=dataframe['evaluation_A'] - dataframe['evaluation_B'])

        dataframe = dataframe.assign(mean=np.mean([dataframe.as_matrix(columns=['evaluation_A']),
                                                   dataframe.as_matrix(columns=['evaluation_B'])], axis=0))

        k = len(pd.unique(dataframe.subject))  # number of conditions
        N = len(dataframe.values)  # conditions times participants

        DFbetween = k - 1
        DFwithin = N - k
        DFtotal = N - 1

        anova_data = pd.DataFrame()
        dataframe_summary = dataframe.groupby(['subject'])
        anova_data['count'] = dataframe_summary['diff'].count()  # number of values in each group ng
        anova_data['sum'] = dataframe_summary['diff'].sum()  # sum of values in each group
        anova_data['mean'] = dataframe_summary['diff'].mean()  # mean of values in each group Xg
        anova_data['variance'] = dataframe_summary['diff'].var()
        anova_data['sd'] = np.sqrt(anova_data['variance'])
        anova_data['count_sqr'] = anova_data['count'] ** 2

        grand_mean = anova_data['sum'].sum() / anova_data['count'].sum()  # XG

        # Calculate the MSS within
        squared_within = 0
        for name, group in dataframe_summary:
            group_mean = group['diff'].sum() / group['diff'].count()

            squared = 0
            for index, row in group.iterrows():
                squared += (row['diff'] - group_mean) ** 2

            squared_within += squared

        SSwithin = squared_within

        # Calculate the MSS between
        ss_between_partial = 0
        for index, row in anova_data.iterrows():
            ss_between_partial += row['count'] * ((row['mean'] - grand_mean) ** 2)

        SSbetween = ss_between_partial

        #  Calculate SS total
        squared_total = 0
        for index, row in dataframe.iterrows():
            squared_total += (row['diff'] - grand_mean) ** 2

        SStotal = squared_total

        MSbetween = SSbetween / DFbetween
        MSwithin = SSwithin / DFwithin

        n = DFbetween + 1
        m = DFtotal + 1
        sigma_m2 = sum(anova_data['count_sqr'])

        variance_b_method = MSwithin

        diff_bet_within = MSbetween - MSwithin
        divisor = (m ** 2 - sigma_m2) / ((n - 1) * m)
        variance = diff_bet_within / divisor

        total_variance = variance + variance_b_method
        sd = np.sqrt(total_variance)

        mean_bias = sum(anova_data['sum']) / m
        upper_loa = mean_bias + (1.96 * sd)
        lower_loa = mean_bias - (1.96 * sd)

        return dataframe, mean_bias, upper_loa, lower_loa

    @staticmethod
    def plot(plot_number, plot_title, x_values, y_values, upper_loa, mean_bias, lower_loa, x_label, y_label):

        x_lim = (2, 8)
        y_lim = (-0.25, 0.3)
        x_annotate_begin = 10.4
        y_gap = 0.05
        ratio_suffix = ''

        plt.figure(plot_number)
        plt.title(plot_title)
        plt.scatter(x_values, y_values)

        plt.axhline(upper_loa, color='gray', linestyle='--')
        plt.axhline(mean_bias, color='gray', linestyle='--')
        plt.axhline(lower_loa, color='gray', linestyle='--')

        plt.annotate(str(BlandAltmanAnalysis._get_antilog(upper_loa))+ratio_suffix, xy=(x_annotate_begin, (upper_loa + y_gap)))
        plt.annotate(str(BlandAltmanAnalysis._get_antilog(mean_bias))+ratio_suffix, xy=(x_annotate_begin, (mean_bias + y_gap)))
        plt.annotate(str(BlandAltmanAnalysis._get_antilog(lower_loa))+ratio_suffix, xy=(x_annotate_begin, (lower_loa + y_gap)))

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show()
