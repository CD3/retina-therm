import itertools
import os
import time

from retina_therm.parallel_jobs import *


def test_parallel_job_controller():
    class MyProcess(JobProcessorBase):
        def run_job(self, config):
            time.sleep(1)
            self.progress.emit(1, 1)
            return config

    controller = BatchJobController(MyProcess, njobs=1)
    controller.start()
    start = time.perf_counter()
    controller.run_jobs([1, 2])
    end = time.perf_counter()
    controller.stop()
    controller.wait()

    assert end - start > 2
    # assert len(controller.results) == 1
    # assert len(controller.results[0]) == 2

    controller = BatchJobController(MyProcess, njobs=2)
    controller.start()

    start = time.perf_counter()
    controller.run_jobs([1, 2])
    end = time.perf_counter()
    controller.stop()
    controller.wait()

    assert end - start > 1
    assert end - start < 1.5
    # assert len(controller.results) == 1
    # assert len(controller.results[0]) == 2


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
