import os


class SignalConnection:
    """A class for encapulating a signal/slot connection."""

    def __init__(self, signal, _id):
        self.__signal = signal
        self.__id = _id

    def disconnect(self):
        self.__signal.disconnect(self.__id)

    def pause(self):
        self.__signal.pause(self.__id)

    def unpause(self):
        self.__signal.unpause(self.__id)


class Signal:
    """A signal used to call zero or more functions. NOT meant to be thread-safe."""

    def __init__(self):
        self.clear_slots()

    def connect(self, func):
        _id = id(func)
        self.__slots[_id] = func
        return SignalConnection(self, _id)

    def pause(self, _id):
        self.__paused.add(_id)

    def unpause(self, _id):
        self.__paused.discard(_id)

    def disconnect(self, _id):
        try:
            del self.__slots[_id]
        except:
            pass

    def clear_slots(self):
        self.__slots = dict()  # Python 3.7 declates dict() to be _ordered_
        self.__paused = set()

    def num_slots(self):
        return len(self.__slots)

    def emit(self, *args, **kwargs):
        self(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # note that we don't do any checking of the signature here.
        # slots are responsible for accepting the arguments that a signal
        # might pass
        for _id in self.__slots:
            if _id not in self.__paused:
                self.__slots[_id](*args, **kwargs)


class CheckedSignal(Signal):
    """A signal that checks function signatures."""

    pass
