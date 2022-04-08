from typing import List
import pandas as pd

def dicts2df(strucs : List[dict],
             uid : str,
             columns : List[str]) -> pd.DataFrame:
    strucs_transformed = [(s[uid], [s[c] for c in columns]) for s in strucs]
    labels = [t[0] for t in strucs_transformed]
    features = [t[1] for t in strucs_transformed]
    df = pd.DataFrame.from_records(data=features,
                                   index=labels,
                                   columns=columns)
    return df
