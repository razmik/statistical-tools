import pandas as pd
import bland_altman as BA


if __name__ == '__main__':

    data_filename = 'data/BA_Adj_2007_Sample_Data.csv'
    dataframe = pd.read_csv(data_filename)

    dataframe, mean_bias, upper_loa, lower_loa = BA.BlandAltmanAnalysis.analyse(dataframe)

    BA.BlandAltmanAnalysis.plot(1, 'Bland-Altman Analysis', dataframe['mean'], dataframe['diff'],
                                upper_loa, mean_bias, lower_loa,
                                'Difference, RV - IC', 'Average of RV and IC')
