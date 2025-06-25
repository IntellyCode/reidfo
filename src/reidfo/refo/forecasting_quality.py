from typing import Union
import pandas as pd


# reviewed
class ForecastingQuality:
    def __init__(self,
                 train_labels: Union[pd.Series, pd.DataFrame],
                 expected_labels: Union[pd.Series, pd.DataFrame],
                 forecasting_labels: Union[pd.Series, pd.DataFrame]):
        """
        Evaluate the quality of a regime forecasting model.

        :param train_labels: Regime labels produced by the regime model on the training set.
        :param expected_labels: True regime labels ex post on the forecasting set.
        :param forecasting_labels: Regime labels predicted by the forecasting model.
        """
        self.train_labels = train_labels.copy()
        self.expected_labels = expected_labels.copy()
        self.forecasting_labels = forecasting_labels.copy()

        self._validate_inputs()

        index = self.expected_labels.index if isinstance(self.expected_labels, pd.DataFrame) else [0]
        self.evaluation = pd.DataFrame(
            columns=["Model Accuracy", "MCR Accuracy", "Random Accuracy"],
            index=index
        )

    def _validate_inputs(self) -> None:
        types = {type(self.train_labels), type(self.expected_labels), type(self.forecasting_labels)}
        if len(types) > 1:
            raise TypeError("All passed labels must be of the same type (either Series or DataFrame).")

        if isinstance(self.train_labels, pd.DataFrame):
            if not (self.train_labels.index.equals(self.expected_labels.index) and
                    self.train_labels.index.equals(self.forecasting_labels.index)):
                raise ValueError("All DataFrames must have the same row index (e.g. asset IDs or series IDs).")
            if not (self.forecasting_labels.columns.equals(self.expected_labels.columns)):
                raise ValueError("The columns for forecasting and expected labels must be the same")
        elif not isinstance(self.train_labels, pd.Series):
            raise TypeError("Labels must be of type pd.Series or pd.DataFrame.")
        elif not (self.forecasting_labels.index.equals(self.expected_labels.index)):
            raise ValueError("The index for forecasting and expected labels must be the same")

    def _most_common_label(self, train: pd.Series, expected: pd.Series) -> float:
        most_common = train.value_counts().idxmax()
        return (expected == most_common).mean()

    def _random_accuracy(self, train: pd.Series, expected: pd.Series) -> float:
        p_train = train.value_counts(normalize=True).sort_index()
        p_expected = expected.value_counts(normalize=True).sort_index()
        regimes = p_train.index.intersection(p_expected.index)
        return sum(p_train[r] * p_expected[r] for r in regimes)

    def _model_accuracy(self, expected: pd.Series, forecasted: pd.Series) -> float:
        return (expected == forecasted).mean()

    def get_forecasting_stats(self) -> pd.DataFrame:
        """
        Compute model accuracy, MCR accuracy, and random accuracy.

        :return: A DataFrame with metrics per time series (or one row for Series input).
        """
        if isinstance(self.expected_labels, pd.Series):
            acc = self._model_accuracy(self.expected_labels, self.forecasting_labels)
            mcr = self._most_common_label(self.train_labels, self.expected_labels)
            rnd = self._random_accuracy(self.train_labels, self.expected_labels)
            self.evaluation.loc[0] = [acc, mcr, rnd]

        else:
            for row in self.train_labels.index:
                acc = self._model_accuracy(self.expected_labels.loc[row],
                                           self.forecasting_labels.loc[row])
                mcr = self._most_common_label(self.train_labels.loc[row],
                                              self.expected_labels.loc[row])
                rnd = self._random_accuracy(self.train_labels.loc[row],
                                            self.expected_labels.loc[row])
                self.evaluation.loc[row] = [acc, mcr, rnd]

        return self.evaluation
