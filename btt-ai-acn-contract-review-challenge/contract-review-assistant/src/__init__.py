"""
contract-review-assistant source package.

Modules:
    data_utils    — loading, normalization, train/val/test splitting
    chunking      — sentence and paragraph windowing strategies
    model         — clause detection model wrapper
    risk_scoring  — four-signal risk scoring layer
    pipeline      — end-to-end: contract text in → clause register out
"""
