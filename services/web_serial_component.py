"""
Streamlit custom component wrapper for Browser Web Serial hardware connection.

Enables deployed web applications (e.g. on Render HTTPS) to connect to a user's
local Arduino over USB using the browser's Web Serial API.
"""

import os
from typing import Dict, Any, Optional
import streamlit.components.v1 as components

# Path to the HTML/JS component bundle
_COMPONENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "components",
    "web_serial",
)

# Declare custom Streamlit component
_web_serial_component = components.declare_component(
    "web_serial_connector",
    path=_COMPONENT_DIR,
)


def render_web_serial_connector(
    lcd_command: str = "",
    pause_telemetry: bool = False,
    key: str = "web_serial_hw_connector",
) -> Optional[Dict[str, Any]]:
    """
    Renders the browser-side Web Serial hardware connector.

    Args:
        lcd_command: Outgoing serial command to send to Arduino (e.g. "LCD:Line1|Line2\n").
        pause_telemetry: Pause browser-to-Streamlit telemetry updates during analysis.
        key: Streamlit component unique key.

    Returns:
        Dict of latest telemetry from browser Web Serial, e.g.:
        {
            "connected": True,
            "temperature_c": 24.5,
            "humidity_percent": 65.0,
            "gas_value": 312,
            "sensor_status": "active",
            "source": "browser_serial"
        }
        or None if not yet initialized.
    """
    return _web_serial_component(
        lcd_command=lcd_command,
        pause_telemetry=pause_telemetry,
        key=key,
        default=None,
    )
