from src.diagnostics import check_outliers


def test_check_outliers():
    data = {
        "feature_1": [1, 2, 3, 4, 100],
        "target": [0, 1, 0, 1, 0]
    }

    import pandas as pd

    df = pd.DataFrame(data)

    result = check_outliers(df, "target")

    assert "feature_1" in result
    assert result["feature_1"] == 1
    
from src.diagnostics import check_missing_values


def test_check_missing_values():
    data = {
        "feature_1": [1, 2, None, 4, 5],
        "target": [0, 1, 0, 1, 0]
    }
    import pandas as pd
    df = pd.DataFrame(data)
    result = check_missing_values(df)
    assert "feature_1" in result
    assert result["feature_1"] == 1
    
from src.diagnostics import cap_outliers
def test_cap_outliers():
    data = {
        "feature_1": [1, 2, 3, 4, 100],
        "target": [0, 1, 0, 1, 0]
    }
    import pandas as pd
    df = pd.DataFrame(data)
    result = cap_outliers(df, "target")
    assert result["feature_1"].max() < 100