import itertools
import os
import time

from retina_therm.parallel_jobs import *


def test_parallel_job_controller():
    class MyProcess(JobProcess):
        def run_job(self, config):
            time.sleep(1)
            self.progress.emit(1, 1)
            return config

    controller = BatchJobController(MyProcess, 1)
    start = time.perf_counter()
    controller.run([1, 2])
    end = time.perf_counter()
    controller.stop()

    assert end - start > 2
    assert len(controller.results) == 1
    assert len(controller.results[0]) == 2

    controller = BatchJobController(MyProcess, 2)
    start = time.perf_counter()
    controller.run([1, 2])
    end = time.perf_counter()
    controller.stop()

    assert end - start > 1
    assert end - start < 1.5
    assert len(controller.results) == 2
    assert len(controller.results[0]) == 1


def test_parallel_job_controller_and_subjob_controller():
    class MySubProcess(JobProcess):
        def run_job(self, config):
            time.sleep(1)
            self.progress.emit(1, 1)
            return config

    class MyProcess(JobProcess):
        def run_job(self, config):
            num_sub_jobs = len(config)
            controller = BatchJobController(MySubProcess, num_sub_jobs)

            self.current_total_progress = 0

            def compute_progress(msg):
                prog = msg["progress"][0] / msg["progress"][1]
                self.current_total_progress += prog
                return [self.current_total_progress / num_sub_jobs, 1]

            controller.progress.connect(
                lambda msg: self.progress.emit(*compute_progress(msg))
            )
            controller.run(config)
            controller.stop()

            results = list(itertools.chain(*controller.results))

            return results

    controller = BatchJobController(MyProcess, 1)
    start = time.perf_counter()
    controller.run([[1, 2], [3, 4]])
    end = time.perf_counter()
    controller.stop()

    assert end - start > 2
    assert len(controller.results) == 1
    assert len(controller.results[0]) == 2
    assert len(controller.results[0][0]) == 2

    controller = BatchJobController(MyProcess, 2)
    start = time.perf_counter()
    controller.run([[1, 2], [3, 4]])
    end = time.perf_counter()
    controller.stop()

    assert end - start > 1
    assert end - start < 1.5
    assert len(controller.results) == 2
    assert len(controller.results[0]) == 1
    assert len(controller.results[0][0]) == 2
