import pandas as pd
from random import choices


class Filler:
    """
    The class contains functions designed to fill nans and outliers,
    which can be determined by the interquartile range or by the 3-sigma rule.

    It also contains the author's method for filling NaN and outliers - RDBR
    (Random Distribution By Ratio). The bottom line is this:
    the percentage of each normal value (not NaN or outliers) will be
    the probability of filling in that value instead of a specific NaN
    or outlier. In theory, this should help evenly fill in any distribution.

    Input:
      data -  pd.DataFrame;
      columns - pd.Index, list or tuple.
    Output:
      Instance of class.
    """

    def __init__(self, data: pd.DataFrame = None, columns: pd.Index | list | tuple = None) -> None:
        # Errors
        if type(data) != type(pd.DataFrame()):
            raise TypeError(
                "Required positional argument 'data' must be pd.DataFrame")

        if type(columns) not in [type(pd.Index([])), list, tuple]:
            raise TypeError(
                "Required positional argument 'columns' must be pd.Index,"
                + " list or tuple")

        # Attributes
        self.data = data.copy(deep=True)
        self.columns = columns

    def fillnan(self, method: str | int | float = 'rdbr') -> pd.DataFrame:
        """
    This method fill in the NaN.

    Input:
      method - NaN filling method takes any int and float,
               or only 'mean', 'median', 'mode' and 'rdbr'.
    Output:
      pd.Dataframe.
    """

        # Errors
        if type(method) not in [str, int, float]:
            raise TypeError(
                "Required positional argument 'method' must be str, int or float")

        if type(method) == str:
            if method not in ['mean', 'median', 'mode', 'rdbr']:
                raise ValueError(
                    "If Required positional argument 'method' is str, then it accepts"
                    + " only one of these values: 'mean', 'median', 'mode' or 'rdbr'")

        # Code
        for c in self.columns:
            nans = self.data[c].isna()

            if type(method) == int or type(method) == float:
                self.data.loc[nans, c] = method

            if method == 'mean':
                self.data.loc[nans, c] = self.data[c].mean()

            if method == 'median':
                self.data.loc[nans, c] = self.data[c].median()

            if method == 'mode':
                self.data.loc[nans, c] = self.data[c].mode()[0]

            if method == 'rdbr':
                ratio = self.data[c].value_counts(normalize=True)
                rdbr = ([choices(ratio.index, ratio.values)
                         for _ in range(nans.sum())])
                self.data.loc[nans, c] = rdbr

        return self.data

    def filloutlier_iqr(self,
                        method: str | int | float = 'rdbr',
                        k: int | float = 1.5) -> pd.DataFrame:
        """
    This method fill in the outliers by the interquartile range.

    Input:
      method - outliers filling method takes any int and float,
               or only 'mean', 'median' and 'rdbr';
      k - interquartile range multiplier takes any int or float.
    Output:
      pd.Dataframe.
    """

        # Errors
        if type(method) not in [str, int, float]:
            raise TypeError(
                "Required positional argument 'method' must be str, int or float")

        if type(k) not in [int, float]:
            raise TypeError(
                "Required positional argument 'k' must be int or float")

        if type(method) == str:
            if method not in ['mean', 'median', 'rdbr']:
                raise ValueError(
                    "If Required positional argument 'method' is str, then it accepts"
                    + " only one of these values: 'mean', 'median' or 'rdbr'")

        # Code
        for c in self.columns:
            q1, q3 = self.data[c].quantile(0.25), self.data[c].quantile(0.75)
            iqr = k * (q3 - q1)
            left, right = q1 - iqr, q3 + iqr
            outliers = (self.data[c] < left) | (self.data[c] > right)
            no_outliers = self.data.loc[~outliers, c]

            if type(method) == int or type(method) == float:
                self.data.loc[outliers, c] = method

            if method == 'mean':
                self.data.loc[outliers, c] = no_outliers.mean()

            if method == 'median':
                self.data.loc[outliers, c] = no_outliers.median()

            if method == 'rdbr':
                ratio = no_outliers.value_counts(normalize=True)
                rdbr = ([choices(ratio.index, ratio.values)
                         for _ in range(self.data.loc[outliers, c].size)])
                self.data.loc[outliers, c] = rdbr

        return self.data

    def filloutlier_sigma(self,
                          method: str | int | float = 'rdbr',
                          k: int | float = 3) -> pd.DataFrame:
        """
    This method fill in the outliers by the 3-sigma rule.

    Input:
      method - outliers filling method takes any int and float,
               or only 'mean', 'median' and 'rdbr';
      k - sigma multiplier takes any int or float.
    Output:
      pd.Dataframe.
    """

        # Errors
        if type(method) not in [str, int, float]:
            raise TypeError(
                "Required positional argument 'method' must be str, int or float")

        if type(k) not in [int, float]:
            raise TypeError(
                "Required positional argument 'k' must be int or float")

        if type(method) == str:
            if method not in ['mean', 'median', 'rdbr']:
                raise ValueError(
                    "If Required positional argument 'method' is str, then it accepts"
                    + " only one of these values: 'mean', 'median' or 'rdbr'")

        # Code
        for c in self.columns:
            mean, sigma = self.data[c].mean(), k * self.data[c].std()
            left, right = mean - sigma, mean + sigma
            outliers = (self.data[c] < left) | (self.data[c] > right)
            no_outliers = self.data.loc[~outliers, c]

            if type(method) == int or type(method) == float:
                self.data.loc[outliers, c] = method

            if method == 'mean':
                self.data.loc[outliers, c] = no_outliers.mean()

            if method == 'median':
                self.data.loc[outliers, c] = no_outliers.median()

            if method == 'rdbr':
                ratio = no_outliers.value_counts(normalize=True)
                rdbr = ([choices(ratio.index, ratio.values)
                         for _ in range(self.data.loc[outliers, c].size)])
                self.data.loc[outliers, c] = rdbr

        return self.data
