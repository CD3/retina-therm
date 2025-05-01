import tqdm


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
