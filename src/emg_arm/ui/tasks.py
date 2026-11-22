"""Run a blocking function off the GUI thread and get the result back as a signal."""

from __future__ import annotations

from PyQt5.QtCore import QObject, QThread, pyqtSignal

_alive = set()


class Task(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:          # message is shown to the user
            self.failed.emit(str(exc) or exc.__class__.__name__)
            return
        self.done.emit(result)


def run_task(fn, *args, on_done=None, on_failed=None, on_progress=None, parent=None, **kwargs) -> Task:
    """``fn`` may accept a ``progress`` keyword; it is passed a callable that emits ``Task.progress``."""
    task = Task(fn, *args, parent=parent, **kwargs)
    if on_progress is not None:
        task._kwargs["progress"] = task.progress.emit
        task.progress.connect(on_progress)
    if on_done:
        task.done.connect(on_done)
    if on_failed:
        task.failed.connect(on_failed)
    _alive.add(task)
    task.finished.connect(lambda: _alive.discard(task))
    task.start()
    return task
