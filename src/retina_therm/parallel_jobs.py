import asyncio
import multiprocessing
import os
import time
from collections import deque

from .progress_display import *
from .signals import *


def pprint(*args):
    print(os.getpid(), *args)


def get_open_file_descriptors(pid):
    fds = set(os.listdir("/proc/self/fd"))
    return fds


def mkmsg(t, p):
    "Create a message of type `t` with payload `p`"
    return {"type": t, "payload": p}


class JobProcessorBase(multiprocessing.Process):
    """
    A class for implementing work that will run in a separate process.

    To use, subclass and implement the `run_job(...)` method.

    To communicate with parent, use `msg_send(msg)`, `mes_poll()`, and `msg_recv()` method.
    """

    def __init__(self):
        super().__init__()
        self.parent_pid = os.getpid()
        self.parent_link, self.child_link = multiprocessing.Pipe()
        self.progress = Signal()
        self.status = Signal()

    def run_job(self, config):
        raise RuntimeError("run_job(...) not implemented")

    def msg_send(self, msg):
        link = self.parent_link if os.getpid() == self.parent_pid else self.child_link
        link.send(msg)

    def msg_recv(self):
        link = self.parent_link if os.getpid() == self.parent_pid else self.child_link
        return link.recv()

    def msg_poll(self):
        link = self.parent_link if os.getpid() == self.parent_pid else self.child_link
        return link.poll()

    def run(self):  # runs in CHILD
        """
        Starts a server to process jobs. Jobs are sent as `call` messages. The
        payload is passed directly to `run_job(job)`, which is implemented by the
        subclass.
        """
        # connect CHILD slots to forward signals as messages to the parent
        self.progress.connect(lambda *args: self.msg_send(mkmsg("progress", args)))
        self.status.connect(lambda msg: self.msg_send(mkmsg("status", msg)))

        running = True
        self._start()
        while running:
            msg = self.msg_recv()
            if msg["type"] == "call":
                if msg["payload"] == "stop":
                    running = False
                    self._stop()
                else:
                    try:
                        result = self.run_job(msg["payload"])
                        self.msg_send(mkmsg("result", result))
                        self.msg_send(mkmsg("reply", "finished"))
                    except Exception as e:
                        self.msg_send(mkmsg("error", str(e)))
        self.progress.clear_slots()
        self.status.clear_slots()

    def _start(self):
        """
        Derived classes can implement to run any setup code that is needed.
        """
        pass

    def _stop(self):
        """
        Derived classes can implement to run any teardown code that is needed.
        """
        pass


class BatchJobController:
    def __init__(self, proc_type, *, njobs, args={}):
        self.parent_pid = os.getpid()
        self.processes = list(
            map(lambda t: t(**args), [proc_type] * njobs)
        )  # list of process instances
        self.progress = Signal()
        self.status = Signal()

    def start(self):
        deque(map(lambda p: p.start(), self.processes))

    def stop(self):
        deque(
            map(
                lambda p: p.msg_send(mkmsg("call", "stop")),
                self.processes,
            )
        )

    def wait(self):
        deque(map(lambda p: p.join(), self.processes))

    def run_jobs(self, jobs):
        """
        Run jobs in subprocesses. Results will be returned in order (in a list) even though
        the jobs to not have to finish in order.
        """
        # running is a list that stores the job number running in each process. -1 means "no job running".
        # results is a list of results returned by the processes that run a job. it is "ordered".
        running = [-1] * len(self.processes)
        results = [None] * len(jobs)
        # if no jobs are running, then all elements of the running list will be -1
        # and sum(running) will be -(number of processes)
        while len(jobs) > 0 or sum(running) > -len(self.processes):
            for i, p in enumerate(self.processes):
                if len(jobs) > 0 and running[i] < 0:
                    # there is a job to run and this process is not running anything
                    p.msg_send(mkmsg("call", jobs.pop()))
                    running[i] = len(jobs)
                if p.msg_poll():
                    msg = p.msg_recv()
                    if msg["type"] == "result":
                        results[running[i]] = msg["payload"]
                    elif msg["type"] == "reply":
                        if msg["payload"] == "finished":
                            running[i] = -1
                    elif msg["type"] == "progress":
                        self.progress.emit(i, msg["payload"])
                    elif msg["type"] == "status":
                        self.status.emit(i, msg["payload"])
                    elif msg["type"] == "error":
                        print("There was an exception in the in the child process")
                        print(msg["payload"])

                    else:
                        raise RuntimeError(f"Unknown message type, msg: {msg}")
        return results
