import asyncio
import multiprocessing

import tqdm

from .signals import *


class JobProcess(multiprocessing.Process):
    """A class for running jobs in a separate process."""

    def run_job(self, config):
        # This method needs to be overridden by derived class to perform the actual
        # work to be done. It will be ran in a separate process.
        raise TypeError(f"run_job(...) method MUST be implemented by derived class.")

    def __init__(self):
        super().__init__()
        # create pipe to communicate between child and parent procs
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.parent_conn_progress, self.child_conn_progress = multiprocessing.Pipe()
        self.parent_conn_status, self.child_conn_status = multiprocessing.Pipe()
        self.control = Signal()
        self.progress = Signal()
        self.status = Signal()

    def run(self):  # REQUIRED: we must implement this to hook into multiprocessing
        # this runs in the child process
        active = True
        while active:
            msg = self.child_conn.recv()
            if msg == "quit":
                active = False
            else:
                try:
                    self.run_job(msg)
                    self.child_conn.send("finished")
                except Exception as e:
                    self.child_conn_status.send(
                        f"ERROR: exception thrown while running job: {e}."
                    )
                    self.child_conn.send("finished")

    def submit_job(self, config):
        self.parent_conn.send(config)

    async def process_jobs(self, config_queue):
        # this runs in the parent process
        self.shutdown = False
        self.busy = False
        # NOTE: order seems to matter here.
        # tasks will be scheduled in the order they are submitted?
        ph = asyncio.create_task(self.handle_progress_messages())
        sh = asyncio.create_task(self.handle_status_messages())
        ch = asyncio.create_task(self.handle_control_messages())

        while len(config_queue):
            # BUG: don't wait here
            # while self.busy:
            #     await asyncio.sleep(0)
            self.submit_job(config_queue.pop())
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
                await asyncio.sleep(0)

        self.shutdown = True
        self.parent_conn.send("quit")

        await ph
        await sh
        await ch

    async def handle_control_messages(self):
        while not self.shutdown:
            if self.parent_conn.poll():
                msg = self.parent_conn.recv()
                self.control.emit(msg)
                if msg == "finished":
                    self.busy = False
            await asyncio.sleep(0)  # this is how we yield control to the scheduler

    async def handle_progress_messages(self):
        while not self.shutdown:
            if self.parent_conn_progress.poll():
                prog = self.parent_conn_progress.recv()
                self.progress.emit(*prog)
            await asyncio.sleep(0)  # this is how we yield control to the scheduler

    async def handle_status_messages(self):
        while not self.shutdown:
            if self.parent_conn_status.poll():
                stat = self.parent_conn_status.recv()
                self.status.emit(stat)
            await asyncio.sleep(0)


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
        if "ERROR" in text:
            tqdm.tqdm.write(text)


class Controller:
    """A class for setting up and monitoriing multiple processes to run batch jobs."""

    def __init__(self, job_process_type: JobProcess, num_job_processes=None):
        self.num_job_processes = (
            num_job_processes if num_job_processes else multiprocessing.cpu_count()
        )

        self.display = ProgressDisplay()
        self.job_procs = [job_process_type() for i in range(self.num_job_processes)]

        self.display.setup_new_bar("Total")
        for p in self.job_procs:
            # note that our lambdas need to define a defaulted arg (_id here)
            # so that they capture the value of each procs name
            self.display.setup_new_bar(p.name)
            p.progress.connect(
                lambda i, n, _id=p.name: self.display.set_progress(_id, i, n)
            )
            p.control.connect(
                lambda msg, _id=p.name: (
                    self.display.set_progress(_id, 1, 1) if msg == "finished" else None
                )
            )
            p.control.connect(
                lambda msg: (
                    self.display.update_progress("Total") if msg == "finished" else None
                )
            )
            p.status.connect(lambda msg: self.display.print(msg))
            p.start()
        self.event_loop = asyncio.get_event_loop()

    def run(self, configs):
        self.display.set_total("Total", len(configs))

        tasks = [
            self.event_loop.create_task(p.process_jobs(configs)) for p in self.job_procs
        ]
        self.event_loop.run_until_complete(asyncio.gather(*tasks))
        for t in tasks:
            t.cancel()

    def stop(self):
        for p in self.job_procs:
            p.join()

        self.display.close()
