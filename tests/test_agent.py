import pytest
from agent_test.geo_agent import geo_agent


@pytest.mark.parametrize(
    "query,ideal",
    [
        ("What is the temperature at 35.97583846 and long=-84.2743123", None),
        ("What is the elevation at 35.97583846 and long=-84.2743123", "293"),
        ("Describe the features you see at 35.97583846 and long=-84.2743123", "lake"),
    ],
)
def test_agent(query, ideal):
    r = geo_agent.run_sync(query)
    data = r.data
    print(data)
    assert data is not None
    if ideal is not None:
        if isinstance(ideal, str):
            assert ideal.lower() in data.lower()
        elif isinstance(ideal, int):
            assert ideal == data
        elif isinstance(ideal, float):
            assert abs(ideal - data) < 0.1
    print("TEST RESULT:", data)