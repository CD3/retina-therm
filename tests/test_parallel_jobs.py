import os
import time

from retina_therm.parallel_jobs import *


def test_parallel_job_controller():
    class MyProcess(JobProcess):
        def run_job(self, config):
            time.sleep(1)

    controller = Controller(MyProcess, 1, no_progress_display=False)
    start = time.perf_counter()
    controller.run([1, 2])
    end = time.perf_counter()
    controller.stop()

    assert end - start > 1
    assert end - start > 2

    controller = Controller(MyProcess, 2, no_progress_display=True)
    start = time.perf_counter()
    controller.run([1, 2])
    end = time.perf_counter()
    controller.stop()

    assert end - start > 1
    assert end - start < 1.5
