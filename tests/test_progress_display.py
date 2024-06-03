import pytest

from retina_therm.parallel_jobs import *


def test_adding_bars():
    display = ProgressDisplay()

    display.setup_new_bar("Bar-1")
    display.setup_new_bar("Bar-2")

    assert len(display.bars) == 2
    assert len(display.totals) == 2
    assert len(display.iters) == 2
    assert "Bar-1" in display.totals and display.totals["Bar-1"] is None
    assert "Bar-2" in display.totals and display.totals["Bar-2"] is None
    display.close()


def test_bar_totals():
    display = ProgressDisplay()

    display.setup_new_bar("Bar-1")
    display.setup_new_bar("Bar-2", total=4)

    assert len(display.bars) == 2
    assert len(display.totals) == 2
    assert len(display.iters) == 2
    assert "Bar-1" in display.totals and display.totals["Bar-1"] is None
    assert "Bar-2" in display.totals and display.totals["Bar-2"] is 4

    display.set_progress("Bar-1", 1, 2)
    display.set_progress("Bar-2", 1)

    with pytest.raises(RuntimeError) as e:
        display.set_progress("Bar-1", 1)
    assert "Could not determine total number of iteration" in str(e)
    assert "Bar-1" in str(e)

    with pytest.raises(RuntimeError) as e:
        display.set_total("missing", 1)
    assert "No bar tagged" in str(e)
    assert "has been setup" in str(e)
    assert "missing" in str(e)

    display.close()


def test_bar_iterations():
    display = ProgressDisplay()

    display.setup_new_bar("Bar-1")
    display.setup_new_bar("Bar-2", total=4)

    display.set_progress("Bar-1", 1, 10)
    display.set_progress("Bar-2", 2)

    assert display.iters["Bar-1"] == 1
    assert display.iters["Bar-2"] == 2

    display = ProgressDisplay()

    display.setup_new_bar("Bar-1")
    display.setup_new_bar("Bar-2", total=4)

    with pytest.raises(RuntimeError) as e:
        display.update_progress("Bar-1")
    assert "Could not determine total number of iteration" in str(e)
    display.update_progress("Bar-2")
