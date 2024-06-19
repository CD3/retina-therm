from unittest.mock import MagicMock, Mock

import pytest

from retina_therm.signals import *


class Mock:
    pass


def test_basic_usage():
    a = Signal()

    slot_tracker = Mock()
    slot_tracker.callback = MagicMock()

    def callback(v):
        slot_tracker.callback(v)

    a.connect(callback)

    a.emit(1)
    slot_tracker.callback.assert_called_with(1)


def test_multiple_slots():
    a = Signal()

    slot_tracker = Mock()
    slot_tracker.callback1 = MagicMock()
    slot_tracker.callback2 = MagicMock()

    def callback1(v):
        slot_tracker.callback1(v)

    def callback2(v):
        slot_tracker.callback2(2 * v)

    a.connect(callback1)
    a.connect(callback2)

    a.emit(1)
    slot_tracker.callback1.assert_called_with(1)
    slot_tracker.callback2.assert_called_with(2)


def test_disconnecting():
    a = Signal()

    slot_tracker = Mock()
    slot_tracker.callback1 = MagicMock()
    slot_tracker.callback2 = MagicMock()

    def callback1(v):
        slot_tracker.callback1(v)

    def callback2(v):
        slot_tracker.callback2(2 * v)

    conn1 = a.connect(callback1)
    conn2 = a.connect(callback2)

    a.emit(1)
    slot_tracker.callback1.assert_called_with(1)
    slot_tracker.callback2.assert_called_with(2)

    conn2.disconnect()
    slot_tracker.callback1.reset_mock()
    slot_tracker.callback2.reset_mock()

    a.emit(2)
    slot_tracker.callback1.assert_called_with(2)
    slot_tracker.callback2.assert_not_called()

    conn2 = a.connect(callback2)
    slot_tracker.callback1.reset_mock()
    slot_tracker.callback2.reset_mock()

    a.emit(3)
    slot_tracker.callback1.assert_called_with(3)
    slot_tracker.callback2.assert_called_with(6)

    conn1.pause()
    slot_tracker.callback1.reset_mock()
    slot_tracker.callback2.reset_mock()

    a.emit(4)
    slot_tracker.callback1.assert_not_called()
    slot_tracker.callback2.assert_called_once_with(8)

    conn1.unpause()
    a.emit(5)
    slot_tracker.callback1.assert_called_once_with(5)
    slot_tracker.callback2.assert_called_with(10)

    conn1.pause()
    conn1.pause()

    a.emit(6)
    slot_tracker.callback1.assert_called_once_with(5)
    slot_tracker.callback2.assert_called_with(12)


def test_forwarding_signals():
    a = Signal()
    b = Signal()

    slot_tracker = Mock()
    slot_tracker.callback = MagicMock()

    def callback(v):
        slot_tracker.callback(v)

    b.connect(callback)
    a.connect(b)

    a.emit(1)
    slot_tracker.callback.assert_called_with(1)


def test_errors_in_slots():
    a = Signal()
    b = Signal()

    slot_tracker = Mock()
    slot_tracker.callback = MagicMock()

    def callback(v):
        # try to call a method that does not exist in the slot
        slot_tracker.callback_typo(v)

    b.connect(callback)
    a.connect(b)

    with pytest.raises(AttributeError) as e:
        a.emit(1)

    assert "no attribute 'callback_typo'" in str(e)
