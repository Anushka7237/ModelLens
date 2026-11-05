from src.recommendations import (
    generate_recommendations,
    create_recommendation_summary
)


def test_generate_recommendations():
    status = {
        "missing_values": True,
        "outliers": True
    }
    results = generate_recommendations(status)
    assert len(results) == 2


def test_recommendations_priority_order():
    status = {
        "outliers": True,
        "target_leakage": True,
        "near_constant_features": True
    }
    results = generate_recommendations(status)
    priorities = [item["priority"] for item in results]
    assert priorities == ["High", "Medium", "Low"]


def test_empty_recommendations():
    results = generate_recommendations({})
    assert results == []


def test_recommendation_summary():
    results = generate_recommendations({
        "missing_values": True,
        "outliers": True
    })
    summary = create_recommendation_summary(results)
    assert summary["total_recommendations"] == 2
    assert summary["high_priority"] == 1
    assert summary["medium_priority"] == 1