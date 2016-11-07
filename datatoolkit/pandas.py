
import pandas as pd
import itertools


def explode(df, explode_col):
    """
    Similar to Hive's "explode" command: verticalize the explode_col of this pandas dataframe, assuming this 
    column contains lists
    """
    
    def explode_row(row):
        return [row.drop(explode_col).append(pd.Series("sim", index=[explode_col]))
                    for p in row[explode_col]]

    rows_list = df.apply(explode_row, axis=1)

    return pd.concat([pd.DataFrame(rows) for rows in rows_list])



def cartesian(df1, df2):
    """
    Return a "cartesian product" of both dataframe, i.e the result is one dataframe with all the 
    columns of df1 + all the columns of df2, and each row of df1 is matched with every row of df2
    """

    rows = itertools.product(df1.iterrows(), df2.iterrows())

    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)