import pandas as pd

from .two_regime_strategy import TwoRegimeStrategy


# reviewed
class LongShortStrategy(TwoRegimeStrategy):
    def __call__(self, returns: pd.Series, labels: pd.Series) -> pd.Series:
        df = self._prepare(returns, labels)
        return df.apply(
            lambda row: row['return'] if row['regime'] == self.green_regime else -row['return'],
            axis=1
        )
