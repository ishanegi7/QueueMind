def test_port_configuration_override():
    """Verify PORT takes precedence over API_PORT."""
    import os
    from api.config import Settings

    # Test default
    settings = Settings()
    assert settings.PORT is None
    assert settings.API_PORT == 8000

    # Test overriding PORT
    os.environ["PORT"] = "8080"
    settings_with_port = Settings()
    assert settings_with_port.PORT == 8080
    assert settings_with_port.API_PORT == 8000

    del os.environ["PORT"]
