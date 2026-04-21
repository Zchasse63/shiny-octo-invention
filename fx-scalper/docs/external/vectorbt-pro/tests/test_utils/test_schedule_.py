import asyncio
import os
from datetime import time as _time

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import datetime_, schedule_


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestScheduleManager:
    def test_every(self):
        manager = schedule_.ScheduleManager()
        job = manager.every()
        assert job.interval == 1
        assert job.unit == "seconds"
        assert job.at_time is None
        assert job.start_day is None

        job = manager.every(10, "minutes")
        assert job.interval == 10
        assert job.unit == "minutes"
        assert job.at_time is None
        assert job.start_day is None

        job = manager.every("hour")
        assert job.interval == 1
        assert job.unit == "hours"
        assert job.at_time is None
        assert job.start_day is None

        job = manager.every("10:30")
        assert job.interval == 1
        assert job.unit == "days"
        assert job.at_time == _time(10, 30)
        assert job.start_day is None

        job = manager.every("day", "10:30")
        assert job.interval == 1
        assert job.unit == "days"
        assert job.at_time == _time(10, 30)
        assert job.start_day is None

        job = manager.every("day", _time(9, 30, tzinfo=datetime_.get_utc_tz()))
        assert job.interval == 1
        assert job.unit == "days"
        assert job.at_time == datetime_.tzaware_to_naive_time(
            _time(9, 30, tzinfo=datetime_.get_utc_tz()),
            datetime_.get_local_tz(),
        )
        assert job.start_day is None

        job = manager.every("monday")
        assert job.interval == 1
        assert job.unit == "weeks"
        assert job.at_time is None
        assert job.start_day == "monday"

        job = manager.every("wednesday", "13:15")
        assert job.interval == 1
        assert job.unit == "weeks"
        assert job.at_time == _time(13, 15)
        assert job.start_day == "wednesday"

        job = manager.every("minute", ":17")
        assert job.interval == 1
        assert job.unit == "minutes"
        assert job.at_time == _time(0, 0, 17)
        assert job.start_day is None

    def test_start(self):
        kwargs = dict(call_count=0)

        def job_func(kwargs):
            kwargs["call_count"] += 1
            if kwargs["call_count"] == 5:
                raise KeyboardInterrupt

        manager = schedule_.ScheduleManager()
        manager.every().do(job_func, kwargs)
        manager.start()
        assert kwargs["call_count"] == 5

    def test_async_start(self):
        kwargs = dict(call_count=0)

        def job_func(kwargs):
            kwargs["call_count"] += 1
            if kwargs["call_count"] == 5:
                raise schedule_.CancelledError

        manager = schedule_.ScheduleManager()
        manager.every().do(job_func, kwargs)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(manager.async_start())
        assert kwargs["call_count"] == 5
