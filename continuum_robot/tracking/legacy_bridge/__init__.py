"""Legacy bridge compatibility helpers.

The active tracking runtime is the Python NDI backend. This package keeps the
old bridge/socket compatibility path isolated so it is explicit that the code is
legacy-only and not part of the normal runtime path.
"""

from .tracker_protocol import TrackerStatusMessage, TrackerTransformMessage, parse_tracker_json_line
from .tracker_service_manager import TrackerServiceManager
from .tracker_socket_client import TrackerSocketClient

__all__ = [
    "TrackerServiceManager",
    "TrackerSocketClient",
    "TrackerStatusMessage",
    "TrackerTransformMessage",
    "parse_tracker_json_line",
]
