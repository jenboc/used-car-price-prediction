import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HierarchicalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, hierarchy, method, mark_missing=False):
        self.columns = columns
        self.hierarchy = hierarchy
        self.method = method
        self.mark_missing = mark_missing

    def decide_method(self):
        if self.method == "mode":
            return lambda x: x.mode()[0]
        elif self.method == "median":
            return lambda x: x.median()

        return lambda x: x

    # Discover the modal categories
    def fit(self, X, y=None):
        X = X.copy()

        # Mapping for each entry in hierarchy list + a global mapping
        self.hierarchy_mappings = [{} for k in self.hierarchy] + [{}]

        method_func = self.decide_method()

        for col in self.columns:
            # Work out mapping for each hierarchy grouping
            for i, k in enumerate(self.hierarchy):
                self.hierarchy_mappings[i][col] = (
                    X.dropna(subset=[col] + k)
                     .groupby(k)[col]
                     .agg(method_func)
                )

            # Global map
            self.hierarchy_mappings[-1][col] = \
                X[col].dropna().pipe(method_func)

        return self

    # Fill in any missing gaps using the modal categories
    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            if self.mark_missing:
                X[f"{col}_missing"] = X[col].isna().astype(int)

            # Descend hierarchy, attempting to fill NaN values
            for hierarchy, mappings in zip(
                    self.hierarchy,
                    self.hierarchy_mappings[:-1]
            ):
                mapping = mappings[col]

                values = None
                if len(hierarchy) == 1:
                    values = X[hierarchy[0]].map(mapping)
                else:
                    keys = pd.MultiIndex.from_frame(X[hierarchy])
                    values = pd.Series(
                        mapping.reindex(keys).values,
                        index=X.index
                    )

                values = values.astype(X[col].dtype)
                X[col] = X[col].fillna(values)

            # Fill remaining with global
            X[col] = X[col].fillna(self.hierarchy_mappings[-1][col])

        return X
