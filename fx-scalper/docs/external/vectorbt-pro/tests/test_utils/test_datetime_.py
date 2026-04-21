import os
from datetime import datetime as _datetime
from datetime import time as _time
from datetime import timedelta as _timedelta
from datetime import timezone as _timezone

import pytest
from pandas.tseries.frequencies import to_offset

import vectorbtpro as vbt
from tests.utils import *
from vectorbtpro._dtypes import *
from vectorbtpro.utils import datetime_


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


class TestDatetime:
    def test_to_offsets(self):
        assert datetime_.to_offset("d") == to_offset("1d")
        assert datetime_.to_offset("day") == to_offset("1d")
        assert datetime_.to_offset("m") == to_offset("1min")
        assert datetime_.to_offset("1m") == to_offset("1min")
        assert datetime_.to_offset("1 m") == to_offset("1min")
        assert datetime_.to_offset("1 minute") == to_offset("1min")
        assert datetime_.to_offset("2 minutes") == to_offset("2min")
        assert datetime_.to_offset("1 hour, 2 minutes") == to_offset("1h 2min")
        assert datetime_.to_offset("1 hour; 2 minutes") == to_offset("1h 2min")
        assert datetime_.to_offset("2 weeks") == pd.offsets.Week(weekday=0) * 2
        assert datetime_.to_offset("2 months") == pd.offsets.MonthBegin() * 2
        assert datetime_.to_offset("2 quarter") == pd.offsets.QuarterBegin(startingMonth=1) * 2
        assert datetime_.to_offset("2 years") == pd.offsets.YearBegin() * 2

    def test_to_timedelta(self):
        assert datetime_.to_timedelta("d") == pd.to_timedelta("1d")
        assert datetime_.to_timedelta("day") == pd.to_timedelta("1d")
        assert datetime_.to_timedelta("m") == pd.to_timedelta("1min")
        assert datetime_.to_timedelta("1m") == pd.to_timedelta("1min")
        assert datetime_.to_timedelta("1 m") == pd.to_timedelta("1min")
        assert datetime_.to_timedelta("1 minute") == pd.to_timedelta("1min")
        assert datetime_.to_timedelta("2 minutes") == pd.to_timedelta("2min")
        assert datetime_.to_timedelta("1 hour, 2 minutes") == pd.to_timedelta("1h 2min")
        assert datetime_.to_timedelta("1 hour; 2 minutes") == pd.to_timedelta("1h 2min")
        assert datetime_.to_timedelta("2 weeks") == pd.Timedelta(days=14)
        assert datetime_.to_timedelta("2 months") == pd.Timedelta(days=365.2425 / 12 * 2)
        assert datetime_.to_timedelta("2 quarter") == pd.Timedelta(days=365.2425 / 4 * 2)
        assert datetime_.to_timedelta("2 years") == pd.Timedelta(days=365.2425 * 2)

    def test_get_utc_tz(self):
        assert datetime_.get_utc_tz().utcoffset(_datetime.now()) == _timedelta(0)

    def test_get_local_tz(self):
        assert datetime_.get_local_tz().utcoffset(_datetime.now()) == _datetime.now().astimezone(None).utcoffset()

    def test_convert_tzaware_time(self):
        assert datetime_.convert_tzaware_time(
            _time(12, 0, 0, tzinfo=datetime_.get_utc_tz()),
            _timezone(_timedelta(hours=2)),
        ) == _time(14, 0, 0, tzinfo=_timezone(_timedelta(hours=2)))

    def test_tzaware_to_naive_time(self):
        assert datetime_.tzaware_to_naive_time(
            _time(12, 0, 0, tzinfo=datetime_.get_utc_tz()),
            _timezone(_timedelta(hours=2)),
        ) == _time(14, 0, 0)

    def test_naive_to_tzaware_time(self):
        assert datetime_.naive_to_tzaware_time(
            _time(12, 0, 0),
            _timezone(_timedelta(hours=2)),
        ) == datetime_.convert_tzaware_time(
            _time(12, 0, 0, tzinfo=datetime_.get_local_tz()),
            _timezone(_timedelta(hours=2)),
        )

    def test_convert_naive_time(self):
        assert datetime_.convert_naive_time(
            _time(12, 0, 0),
            _timezone(_timedelta(hours=2)),
        ) == datetime_.tzaware_to_naive_time(
            _time(12, 0, 0, tzinfo=datetime_.get_local_tz()),
            _timezone(_timedelta(hours=2)),
        )

    def test_is_tz_aware(self):
        assert not datetime_.is_tz_aware(pd.Timestamp("2020-01-01"))
        assert datetime_.is_tz_aware(pd.Timestamp("2020-01-01", tz=datetime_.get_utc_tz()))

    def test_to_timezone(self):
        assert datetime_.to_timezone("utc", to_fixed_offset=True) == _timezone.utc
        assert isinstance(datetime_.to_timezone("Europe/Berlin", to_fixed_offset=True), _timezone)
        assert datetime_.to_timezone("+0500") == _timezone(_timedelta(hours=5))
        assert datetime_.to_timezone(_timezone(_timedelta(hours=1))) == _timezone(_timedelta(hours=1))
        assert datetime_.to_timezone(3600) == _timezone(_timedelta(hours=1))
        assert datetime_.to_timezone(1800) == _timezone(_timedelta(hours=0.5))
        with pytest.raises(Exception):
            datetime_.to_timezone("+05")

    def test_to_tzaware_datetime(self):
        assert datetime_.to_tzaware_datetime(0) == _datetime(1970, 1, 1, 0, 0, 0, 0, tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime(pd.Timestamp("2020-01-01").value, unit="ns") == _datetime(
            2020, 1, 1
        ).replace(tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime("2020-01-01") == _datetime(2020, 1, 1).replace(
            tzinfo=datetime_.get_local_tz()
        )
        assert datetime_.to_tzaware_datetime(pd.Timestamp("2020-01-01")) == _datetime(2020, 1, 1).replace(
            tzinfo=datetime_.get_local_tz()
        )
        assert datetime_.to_tzaware_datetime(pd.Timestamp("2020-01-01", tz=datetime_.get_utc_tz())) == _datetime(
            2020,
            1,
            1,
        ).replace(tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime(_datetime(2020, 1, 1)) == _datetime(2020, 1, 1).replace(
            tzinfo=datetime_.get_local_tz()
        )
        assert datetime_.to_tzaware_datetime(_datetime(2020, 1, 1, tzinfo=datetime_.get_utc_tz())) == _datetime(
            2020,
            1,
            1,
        ).replace(tzinfo=datetime_.get_utc_tz())
        assert datetime_.to_tzaware_datetime(
            _datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime_.get_utc_tz()),
            tz=datetime_.get_local_tz(),
        ) == _datetime(2020, 1, 1, 12, 0, 0, tzinfo=datetime_.get_utc_tz()).astimezone(datetime_.get_local_tz())
        with pytest.raises(Exception):
            datetime_.to_tzaware_datetime("2020-01-001")

    def test_datetime_to_ms(self):
        assert datetime_.datetime_to_ms(_datetime(2020, 1, 1, tzinfo=datetime_.get_utc_tz())) == 1577836800000
