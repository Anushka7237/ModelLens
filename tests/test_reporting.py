from src.reporting import create_report_overview


def test_create_report_overview():
    report = {
        "diagnostics": {
            "status": {
                "outliers": True,
                "missing_values": False
            }
        },
        "explainability": {
            "global_summary": {
                "total_features": 3,
                "top_5_features": ["feature_1", "feature_2", "feature_3"]
            }
        }
    }
    result = create_report_overview(report)
    assert result["total_issues_detected"] == 1
    assert result["detected_issues"] == ["outliers"]
    assert result["total_features"] == 3