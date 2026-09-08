from unittest.mock import MagicMock


run = None
config = MagicMock()


def login(*args, **kwargs):
    return None


def init(*args, **kwargs):
    global run
    run = MagicMock()
    return run


def finish(*args, **kwargs):
    return None


def log(*args, **kwargs):
    return None


def define_metric(*args, **kwargs):
    return None


class Table:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Html:
    def __init__(self, html):
        self.html = html
