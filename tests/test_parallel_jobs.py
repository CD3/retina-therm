import itertools
import math
import os
import time

import pytest

from retina_therm.parallel_jobs import *


def test_parallel_job_processor_simple_usage():
    class SineProcess(JobProcessorBase):
        def run_job(self, x):
            self.status.emit("starting")
            self.progress.emit(0, 1)
            y = math.sin(x)
            self.progress.emit(1, 1)
            self.status.emit("finished")
            return y

    # we can use the process by creating an instance
    p = SineProcess()

    # this will run on the parent process
    assert p.run_job(1) == pytest.approx(math.sin(1))

    # we need to do this in a try incase there is an exception
    # so we can kill the process
    try:
        # to run jobs in the child process we have to start it
        # and send the job as a message.
        p.start()
        p.msg_send(mkmsg("call", 1))

        r = p.msg_recv()
        assert r["type"] == "status"
        assert r["payload"] == "starting"
        r = p.msg_recv()
        assert r["type"] == "progress"
        assert r["payload"] == (0, 1)
        r = p.msg_recv()
        assert r["type"] == "progress"
        assert r["payload"] == (1, 1)
        r = p.msg_recv()
        assert r["type"] == "status"
        assert r["payload"] == "finished"
        r = p.msg_recv()
        assert r["type"] == "result"
        assert r["payload"] == pytest.approx(math.sin(1))

        p.msg_send(mkmsg("call", "stop"))
    except Exception as e:
        p.kill()  # if an exceptoin is thrown we need to make sure
        raise e


def test_parallel_job_processor_parent_vs_child_code():
    class myProcess(JobProcessorBase):
        def run_job(self, arg=None):
            return os.getpid()

    p = myProcess()
    parent_pid = os.getpid()
    assert p.run_job() == parent_pid

    try:
        p.start()
        p.msg_send(mkmsg("call", None))
        r = p.msg_recv()
        assert r["type"] == "result"
        assert r["payload"] != parent_pid

        r = p.msg_recv()
        assert r["type"] == "reply"
        assert r["payload"] == "finished"

        p.msg_send(mkmsg("shutdown", None))

        if p.msg_poll():
            r = p.msg_recv()
            if r["type"] == "error":
                raise RuntimeError(r["payload"])

    except Exception as e:
        p.kill()  # if an exceptoin is thrown we need to make sure
        raise e


def test_parallel_job_processor_context_manager():
    class SineProcess(JobProcessorBase):
        def run_job(self, x):
            y = math.sin(x)
            self.progress.emit(1, 1)
            return y

    with SineProcess() as p:
        # this runs on parent process (the one we are in)
        assert p.run_job(1) == pytest.approx(math.sin(1))

        # this runs on the child process
        p.msg_send(mkmsg("call", 1))
        r = p.msg_recv()
        assert r["type"] == "progress"
        assert r["payload"] == (1, 1)
        r = p.msg_recv()
        assert r["type"] == "result"
        assert r["payload"] == pytest.approx(math.sin(1))

        p.msg_send(mkmsg("shutdown", None))


def test_parallel_job_processor_context_manager_parent_vs_child_procs():
    class myProcess(JobProcessorBase):
        def run_job(self, arg=None):
            return os.getpid()

    parent_pid = os.getpid()
    with myProcess() as p:
        assert p.run_job() == parent_pid

        p.msg_send(mkmsg("call", None))
        r = p.msg_recv()
        assert r["type"] == "result"
        assert r["payload"] != parent_pid

        r = p.msg_recv()
        assert r["type"] == "reply"
        assert r["payload"] == "finished"


def test_parallel_batch_job_controller():
    class MyProcess(JobProcessorBase):
        def run_job(self, config):
            time.sleep(1)
            self.progress.emit(1, 1)
            return config

    try:
        controller = BatchJobController(MyProcess, njobs=1)
        controller.start()
        start = time.perf_counter()
        results = controller.run_jobs([1, 2])
        end = time.perf_counter()
        controller.stop()
        controller.wait()

        assert end - start > 2
        assert len(results) == 2
        assert results[0] == 1
        assert results[1] == 2

        controller = BatchJobController(MyProcess, njobs=2)
        controller.start()

        start = time.perf_counter()
        results = controller.run_jobs([1, 2])
        end = time.perf_counter()
        controller.stop()
        controller.wait()

        assert end - start > 1
        assert end - start < 1.5
        assert len(results) == 2
        assert results[0] == 1
        assert results[1] == 2

        # case were there are more processes than jobs.

        controller = BatchJobController(MyProcess, njobs=3)
        controller.start()

        start = time.perf_counter()
        results = controller.run_jobs([1, 2])
        end = time.perf_counter()
        controller.stop()
        controller.wait()

        assert end - start > 1
        assert end - start < 1.5
        assert len(results) == 2
        assert results[0] == 1
        assert results[1] == 2
    finally:
        controller.kill()


def test_parallel_job_controller_and_subjob_controller():
    class MySubProcess(JobProcessorBase):
        def run_job(self, config):
            time.sleep(1)
            self.progress.emit(1, 1)
            return config

    class MyProcess(JobProcessorBase):
        def run_job(self, config):
            num_sub_jobs = len(config)
            controller = BatchJobController(MySubProcess, njobs=num_sub_jobs)
            controller.start()

            self.current_total_progress = 0

            def compute_progress(msg):
                prog = msg["progress"][0] / msg["progress"][1]
                self.current_total_progress += prog
                return [self.current_total_progress / num_sub_jobs, 1]

            # controller.progress.connect(
            #     lambda msg: self.progress.emit(*compute_progress(msg))
            # )
            results = controller.run_jobs(config)
            controller.stop()
            controller.wait()

            return results

    try:
        controller = BatchJobController(MyProcess, njobs=1)
        controller.start()
        start = time.perf_counter()
        results = controller.run_jobs([[1, 2], [3, 4]])
        end = time.perf_counter()
        controller.stop()
        controller.wait()

        assert end - start > 2
        assert len(results) == 2

        controller = BatchJobController(MyProcess, njobs=2)
        controller.start()
        start = time.perf_counter()
        results = controller.run_jobs([[1, 2], [3, 4]])
        end = time.perf_counter()
        controller.stop()
        controller.wait()

        assert end - start > 1
        assert end - start < 1.5
        assert len(results) == 2
    finally:
        controller.kill()


def test_parallel_job_processor_throwing_exception_in_child():
    class myProcess(JobProcessorBase):
        def run_job(self, arg=None):
            raise RuntimeError("I can't do work")
            return os.getpid()

    with myProcess() as p:
        p.msg_send(mkmsg("call", None))

        r = p.msg_recv()
        assert r["type"] == "exception"
        assert "I can't do work" in r["payload"]
