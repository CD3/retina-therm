import asyncio
import multiprocessing
import os

import tqdm

from .signals import *


def get_open_file_descriptors(pid):
    fds = set(os.listdir("/proc/self/fd"))
    return fds


class JobProcess(multiprocessing.Process):
    """
    A class for running jobs in a separate process.

    This class cannot be used directly, it is a base class. To use it,
    create a subclass and implement the `run_job(self,config)` method.
    This class sets up a job sever that will listen for control commands
    over a multiprocessing pipe and process incoming jobs.
    """

    def run_job(self, config):
        # This method needs to be overridden by derived class to perform the actual
        # work to be done. It will be ran in a separate process, i.e. the child process.
        raise TypeError(f"run_job(...) method MUST be implemented by derived class.")

    def __init__(self):
        super().__init__()
        # create pipes to communicate between child and parent procs
        self.parent_ctrl_conn, self.child_ctrl_conn = multiprocessing.Pipe()
        self.parent_data_conn, self.child_data_conn = multiprocessing.Pipe(duplex=False)

        # create signals that the implementation class can use
        # for communicating with the parent process.
        self.progress = Signal()
        self.status = Signal()
        self.result = Signal()

    def run(self):  # REQUIRED: we must implement this to hook into multiprocessing
        # this runs in the CHILD process
        # it is automatically ran by the multirpocessing library
        #
        # we are basically running a job sever. we get job descriptions
        # (objects) from the parent through the self.child_ctrl_conn pipe and
        # pass the description to the self.run_job(...) method until
        # we recieve a message "quit".
        #
        # setup the progress, status, and result signals to send messages to the parrent proc
        # through the data pipe.
        self.progress.connect(
            lambda i, n: self.child_data_conn.send(
                {"type": "progress", "payload": [i, n]}
            )
        )
        self.status.connect(
            lambda msg: self.child_data_conn.send({"type": "status", "payload": msg})
        )
        self.result.connect(
            lambda obj: self.child_data_conn.send({"type": "result", "payload": obj})
        )

        active = True
        while active:
            # wait for a message containing the job description
            msg = self.child_ctrl_conn.recv()
            if msg == "quit":
                active = False
            else:
                try:
                    result = self.run_job(msg)
                    self.result.emit(result)
                    self.child_ctrl_conn.send("finished")
                except Exception as e:
                    self.status.emit(f"ERROR: exception thrown while running job: {e}.")
                    self.child_ctrl_conn.send("finished")

    async def process_jobs(self, config_queue):
        # this runs in the parent process
        # it is async because we need to monitor several different pipes
        # used to communicate with the child proces.
        self.shutdown = False
        self.busy = False
        # spin up some tasks for handing messages from the child processes.
        # we are using asyncio instead of running them in separate threads.
        # NOTE: order seems to matter here.
        # tasks will be scheduled in the order they are submitted?
        # if so, do we need to make sure that the control messages are handedl _last_
        # so that we don't get a "finished" message from the child and
        # and then shutdown before the result is processed.
        dh = asyncio.create_task(self.handle_data_messages())
        ch = asyncio.create_task(self.handle_ctrl_messages())

        while len(config_queue):
            # make sure we don't wait here
            self.parent_ctrl_conn.send(config_queue.pop())
            self.busy = True

            # while we don't' get any "race conditions"
            # with async code, we can make our own
            # if we call await at the top of the loop above,
            # we are giving control back to the event loop.
            # its possible for another task to pop a job
            # at that point, and then the config_queue will be
            # empty
            # so the _nice_ thing about aync code is, we can
            # be sure anything between `await` statements will be
            # ran syncronously....
            while self.busy:
                await asyncio.sleep(0.1)

        # allow one more call of the handlers
        await asyncio.sleep(0)
        self.shutdown = True
        self.parent_ctrl_conn.send("quit")

        await dh
        await ch

    async def handle_ctrl_messages(self):
        # this runs in the parent process
        # we monitor a pipe for messages from a child
        while not self.shutdown:
            if self.parent_ctrl_conn.poll():
                msg = self.parent_ctrl_conn.recv()
                if msg == "finished":
                    self.busy = False
            await asyncio.sleep(0)  # this is how we yield control to the scheduler

    async def handle_data_messages(self):
        # this runs in the parent process
        while not self.shutdown:
            if self.parent_data_conn.poll():
                msg = self.parent_data_conn.recv()
                if "type" not in msg or msg["type"] not in [
                    "progress",
                    "status",
                    "result",
                ]:
                    raise RuntimeError(
                        "Recieved unknown message type from child over data pipe:", msg
                    )
                if msg["type"] == "progress":
                    self.progress.emit(*msg["payload"])
                if msg["type"] == "status":
                    self.status.emit(msg["payload"])
                if msg["type"] == "result":
                    self.result.emit(msg["payload"])
            await asyncio.sleep(0)  # this is how we yield control to the scheduler


class SilentProgressDisplay:
    """A progress display class that doesn't do anything. Used for "disabling" progress display in the job controllers."""

    def __init__(self):
        pass

    def setup_new_bar(self, tag, total=None):
        pass

    def set_total(self, tag, total):
        pass

    def set_progress(self, tag, i, N=None):
        pass

    def update_progress(self, tag):
        pass

    def close(self):
        pass

    def print(self, text):
        pass


class ProgressDisplay:
    """A class for displaying the progress of multiple jobs."""

    def __init__(self):
        self.bars = dict()
        self.totals = dict()
        self.iters = dict()

    def setup_new_bar(self, tag, total=None):
        self.bars[tag] = tqdm.tqdm(total=100, position=len(self.bars), desc=tag)
        self.totals[tag] = total
        self.iters[tag] = 0

    def set_total(self, tag, total):
        if tag not in self.bars:
            raise RuntimeError(
                f"No bar tagged '{tag}' has been setup. Did you spell it correctly or forget to call setup_new_bar('{tag}')?"
            )
        self.totals[tag] = total

    def set_progress(self, tag, i, N=None):
        if tag not in self.bars:
            self.setup_new_bar(tag)

        if N is None:
            if self.totals[tag] is None:
                raise RuntimeError(
                    f"Could not determine total number of iterations for progress bar. You must either set a total for the tag {tag} with progress_display.set_total('{tag}', TOTAL), or pass the total as an argument, progress_display.set_progress(I, TOTAL)"
                )
            N = self.totals[tag]

        self.iters[tag] = i
        self.bars[tag].n = int(self.bars[tag].total * i / N)
        self.bars[tag].refresh()

    def update_progress(self, tag):
        self.set_progress(tag, self.iters[tag] + 1)

    def close(self):
        for tag in self.bars:
            self.bars[tag].close()

    def print(self, text):
        tqdm.tqdm.write(text)


class BatchJobController:
    """A class for setting up and monitoriing multiple processes to run batch jobs of the same type."""

    def __init__(
        self,
        job_process_type: JobProcess,
        num_job_processes=None,
    ):
        self.num_job_processes = (
            num_job_processes if num_job_processes else multiprocessing.cpu_count()
        )

        # create processes for jobs
        self.job_procs = [job_process_type() for i in range(self.num_job_processes)]
        self.progress = Signal()
        self.results = []

        for i, p in enumerate(self.job_procs):
            # IMPORTANT!!
            # Call p.start() FIRST
            # This is where the fork() happends.
            # Anything before p.start() is called will be copied into the child.
            # we want to connect these slots on the parent process _only_
            p.start()
            self.results.append([])
            p.progress.connect(
                lambda *args, _proc_num=i: self.progress.emit(
                    {"num": _proc_num, "progress": args}
                )
            )
            p.result.connect(
                lambda res, _proc_num=i: self.results[_proc_num].append(res)
            )

        self.event_loop = asyncio.new_event_loop()

    def run(self, configs):
        # we just pass all configurations to all jobs processors.
        # each parent process will pop a job from the queue and send the config
        # to its backend. since python passes lists by reference,
        # the queue will be reduced in all processors when one pops a config.
        # and, since we using async, this will be done in serial so we don't
        # need to worry about races (two processors poping the same job)
        # this makes sure any processes that become available get a job.
        # if we split configs up here it would be possible for one process to
        # get several long running jobs while the others got short jobs,
        # and we would end up waiting on a single proc.
        tasks = [
            self.event_loop.create_task(p.process_jobs(configs)) for p in self.job_procs
        ]
        self.event_loop.run_until_complete(asyncio.gather(*tasks))
        for t in tasks:
            t.cancel()

    def wait(self):
        for p in self.job_procs:
            p.join()

    def stop(self):
        for p in self.job_procs:
            p.parent_ctrl_conn.send("quit")
        self.wait()
