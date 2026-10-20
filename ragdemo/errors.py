class DemoError(Exception):
    """A fixed, safe message that may be shown to the user."""


NO_REFERENCES = "No relevant text or references found."
BLOCKED_REQUEST = "This request was blocked by the document safety policy."
WITHHELD = "The answer was withheld because it could not pass the safety checks."
