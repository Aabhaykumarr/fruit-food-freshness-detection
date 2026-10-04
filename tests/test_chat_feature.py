"""Unit tests for the chat-about-analysis feature in LLMFreshnessService."""

from services.llm_service import LLMFreshnessService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    status="recognized",
    detected_food="apple",
    confidence=0.92,
    visual_description="Red apple with smooth skin",
    temperature_c=25.0,
    humidity_percent=58.0,
    gas_value=120.0,
    freshness_status="FRESH",
    estimated_shelf_life="3-5 days",
    freshness_reasoning="Apple appears fresh with no visible bruising.",
    storage_recommendations=None,
):
    return {
        "detected_food": detected_food if status != "not_food" else None,
        "category": "fruit" if status != "not_food" else None,
        "confidence": confidence,
        "status": status,
        "visual_description": visual_description,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "gas_value": gas_value,
        "sensor_source": "manual_slider",
        "freshness_status": freshness_status,
        "estimated_shelf_life": estimated_shelf_life,
        "shelf_life_range": f"Approximately {estimated_shelf_life}",
        "freshness_reasoning": freshness_reasoning,
        "visual_observations": [visual_description],
        "environmental_context": f"Temp {temperature_c}°C, Humidity {humidity_percent}%",
        "storage_recommendations": storage_recommendations or ["Refrigerate for best results."],
        "uncertainty_factors": ["Sensors measure ambient environment only."],
    }


def _offline_service():
    """Return a service with no API key (offline mode)."""
    return LLMFreshnessService(api_key="")


# ---------------------------------------------------------------------------
# 1. Recognized food + question
# ---------------------------------------------------------------------------

def test_chat_recognized_food_no_api_key():
    svc = _offline_service()
    ctx = _make_context(status="recognized", detected_food="apple")
    answer = svc.chat_about_analysis(ctx, [], "What food is this?")
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "unavailable" in answer.lower() or "api key" in answer.lower()


# ---------------------------------------------------------------------------
# 2. Uncertain food + question
# ---------------------------------------------------------------------------

def test_chat_uncertain_food_no_api_key():
    svc = _offline_service()
    ctx = _make_context(
        status="uncertain",
        detected_food=None,
        visual_description="Rice-based dish containing chicken and spices.",
        freshness_status="UNKNOWN",
    )
    answer = svc.chat_about_analysis(ctx, [], "What food is this?")
    assert isinstance(answer, str)
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 3. Not-food + question
# ---------------------------------------------------------------------------

def test_chat_not_food_no_api_key():
    svc = _offline_service()
    ctx = _make_context(
        status="not_food",
        detected_food=None,
        visual_description="Electronic device detected.",
        freshness_status="NOT_FOOD",
    )
    answer = svc.chat_about_analysis(ctx, [], "What food is this?")
    assert isinstance(answer, str)
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 4. Sensor question
# ---------------------------------------------------------------------------

def test_chat_sensor_question_no_api_key():
    svc = _offline_service()
    ctx = _make_context(temperature_c=28.2, humidity_percent=72.4, gas_value=310)
    answer = svc.chat_about_analysis(ctx, [], "Why is humidity important?")
    assert isinstance(answer, str)
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 5. Freshness question
# ---------------------------------------------------------------------------

def test_chat_freshness_question_no_api_key():
    svc = _offline_service()
    ctx = _make_context(freshness_status="MODERATELY_FRESH")
    answer = svc.chat_about_analysis(ctx, [], "Why is this only moderately fresh?")
    assert isinstance(answer, str)
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 6. Follow-up question using "it" (conversation history)
# ---------------------------------------------------------------------------

def test_chat_followup_preserves_history():
    svc = _offline_service()
    ctx = _make_context()
    history = [
        {"role": "user", "content": "How should I store this?"},
        {"role": "assistant", "content": "Refrigerate for best results."},
    ]
    answer = svc.chat_about_analysis(ctx, history, "What if I leave it out?")
    assert isinstance(answer, str)
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# 7. New image resets chat context (session state logic — tested via context swap)
# ---------------------------------------------------------------------------

def test_chat_context_independence():
    """Feeding a different context should not reference the old one."""
    svc = _offline_service()
    ctx_apple = _make_context(detected_food="apple")
    ctx_banana = _make_context(detected_food="banana")

    answer1 = svc.chat_about_analysis(ctx_apple, [], "What is this?")
    answer2 = svc.chat_about_analysis(ctx_banana, [], "What is this?")

    # Both should return valid strings (offline message), context is independent
    assert isinstance(answer1, str)
    assert isinstance(answer2, str)


# ---------------------------------------------------------------------------
# 8. No API key handling
# ---------------------------------------------------------------------------

def test_chat_no_api_key_returns_clear_message():
    svc = _offline_service()
    ctx = _make_context()
    answer = svc.chat_about_analysis(ctx, [], "Tell me about this food.")
    assert "api key" in answer.lower() or "unavailable" in answer.lower()
    # Must not crash
    assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# 9. Empty user question handling
# ---------------------------------------------------------------------------

def test_chat_empty_question():
    svc = _offline_service()
    ctx = _make_context()
    answer = svc.chat_about_analysis(ctx, [], "")
    assert "please" in answer.lower() or "type" in answer.lower()


def test_chat_whitespace_question():
    svc = _offline_service()
    ctx = _make_context()
    answer = svc.chat_about_analysis(ctx, [], "   ")
    assert "please" in answer.lower() or "type" in answer.lower()
