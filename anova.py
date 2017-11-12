"""
Reference:
http://www.statsmodels.org/dev/generated/statsmodels.stats.anova.anova_lm.html
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols


class ANOVA:

    @staticmethod
    def calculate(dataframe):

        dataframe['mean'] = np.mean(
            [dataframe.as_matrix(columns=['evaluation_A']), dataframe.as_matrix(columns=['evaluation_B'])], axis=0)
        dataframe['diff'] = dataframe['evaluation_A'] - dataframe['evaluation_B']

        mod = ols('diff ~ subject', data=dataframe).fit()

        return sm.stats.anova_lm(mod, typ=2)


