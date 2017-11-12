import pandas as pd
import anova


if __name__ == '__main__':

    data_filename = 'data/BA_Adj_2007_Sample_Data.csv'

    dataframe = pd.read_csv(data_filename)

    anova_table = anova.ANOVA.calculate(dataframe)

    print(anova_table)
