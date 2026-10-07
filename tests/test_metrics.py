import asyncio
import math
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
import pytest
import telegram
from matplotlib.figure import Figure

from api import cbbiinfo_api
from metrics import woobull_topcap_cvdd
from metrics.base_metric import BaseMetric
from metrics.woobull_topcap_cvdd import WoobullMetric


@pytest.fixture
def bitcoin():
    return pl.DataFrame({
        'Date': [
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(12)
        ],
        'PriceLog': [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 2.0, 1.5],
        'PriceHigh': [
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
        ],
        'PriceLow': [
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ],
    })


@pytest.fixture
def axes():
    return list(Figure().subplots(1, 2))


@pytest.fixture
def responses(monkeypatch):
    responses = {}
    transport = httpx.MockTransport(lambda request: responses[str(request.url)])
    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(woobull_topcap_cvdd, 'HTTP', client)
        monkeypatch.setattr(cbbiinfo_api, 'HTTP', client)
        monkeypatch.delenv('TELEGRAM_TOKEN', raising=False)
        monkeypatch.delenv('TELEGRAM_CHAT_ID', raising=False)
        yield responses


def set_chart(responses, bitcoin, *, top, cvdd):
    dates = bitcoin['Date'].dt.epoch(time_unit='ms').to_list()
    responses['https://woocharts.com/bitcoin-price-models/data/chart.json'] = (
        httpx.Response(
            200,
            json={
                'top_': {'x': dates[: len(top)], 'y': top},
                'cvdd': {'x': dates[: len(cvdd)], 'y': cvdd},
            },
        )
    )


def set_fallback(responses, bitcoin, *, name, values):
    dates = bitcoin['Date'].dt.epoch(time_unit='s').to_list()
    responses['https://colintalkscrypto.com/cbbi/data/latest.json'] = httpx.Response(
        200, json={name: dict(zip(dates, values, strict=True))}
    )


@pytest.mark.parametrize('lag', [1, 2], ids=['one-day', 'two-days'])
def test_woobull_lag_still_uses_current_price(bitcoin, axes, responses, lag):
    set_chart(
        responses,
        bitcoin,
        top=[math.exp(4)] * (12 - lag),
        cvdd=[1.0] * (12 - lag),
    )

    values = asyncio.run(WoobullMetric().calculate(bitcoin, axes))

    # Constant models give high=0.725, low=0.25, and current raw value=0.375.
    assert values[-1] == pytest.approx(0.2631578947368421)
    assert values[-2] == pytest.approx(0.5263157894736842)


@pytest.mark.parametrize(
    ('top', 'cvdd', 'expected'),
    [
        ([math.exp(4)] * 10 + [math.exp(5)], [1.0] * 10, 0.10526315789473684),
        ([math.exp(4)] * 10, [1.0] * 10 + [math.exp(0.5)], 0.07518796992481203),
    ],
    ids=['top-arrives-first', 'cvdd-arrives-first'],
)
def test_woobull_preserves_independent_model_updates(
    bitcoin,
    axes,
    responses,
    top,
    cvdd,
    expected,
):
    set_chart(responses, bitcoin, top=top, cvdd=cvdd)

    values = asyncio.run(WoobullMetric().calculate(bitcoin, axes))

    assert values[-1] == pytest.approx(expected)


def test_woobull_fills_internal_holes(bitcoin, axes, responses):
    set_chart(
        responses,
        bitcoin,
        top=[math.exp(4)] * 10 + [None, math.exp(4)],
        cvdd=[1.0] * 10 + [None, 1.0],
    )

    values = asyncio.run(WoobullMetric().calculate(bitcoin, axes))

    assert values[10] == pytest.approx(0.5263157894736842)


def test_woobull_resumes_using_new_observations(bitcoin, axes, responses):
    set_chart(responses, bitcoin, top=[math.exp(4)] * 10, cvdd=[1.0] * 10)
    lagged = asyncio.run(WoobullMetric().calculate(bitcoin, axes))
    set_chart(
        responses,
        bitcoin,
        top=[math.exp(4)] * 10 + [math.exp(5)] * 2,
        cvdd=[1.0] * 12,
    )

    recovered = asyncio.run(WoobullMetric().calculate(bitcoin, axes))

    assert recovered[-1] == pytest.approx(0.10526315789473684)
    assert lagged[:10].equals(recovered[:10])


@pytest.mark.parametrize('failure', ['http-error', 'malformed', 'empty'])
def test_woobull_source_failure_forward_fills_previous_scores(
    bitcoin,
    axes,
    responses,
    failure,
):
    if failure == 'empty':
        set_chart(responses, bitcoin, top=[], cvdd=[])
    else:
        responses['https://woocharts.com/bitcoin-price-models/data/chart.json'] = (
            httpx.Response(503)
            if failure == 'http-error'
            else httpx.Response(200, json={})
        )
    set_fallback(
        responses,
        bitcoin,
        name='Woobull',
        values=[0.37] + [None] * 11,
    )

    values = asyncio.run(WoobullMetric().calculate(bitcoin, axes))

    assert values.to_list() == [0.37] * 12


class StaticMetric(BaseMetric):
    name = 'Static'
    description = 'Static metric'

    def __init__(self, values):
        self.values = pl.Series(values, dtype=pl.Float64)

    def _calculate(self, _df, _ax):
        return self.values


@pytest.mark.parametrize(
    'current',
    [None, math.nan, math.inf, -math.inf],
    ids=['null', 'nan', 'positive-infinity', 'negative-infinity'],
)
def test_invalid_current_value_uses_existing_fallback(
    bitcoin,
    axes,
    responses,
    current,
):
    set_fallback(
        responses,
        bitcoin,
        name='Static',
        values=[0.42] + [None] * 11,
    )

    values = asyncio.run(StaticMetric([0.5] * 11 + [current]).calculate(bitcoin, axes))

    assert values.to_list() == [0.42] * 12


def test_real_zero_and_historical_warmup_do_not_trigger_fallback(
    bitcoin,
    axes,
    responses,
):
    values = asyncio.run(
        StaticMetric([None, math.nan] + [0.0] * 10).calculate(bitcoin, axes)
    )

    assert values[-1] == 0.0
    assert values[0] is None
    assert math.isnan(values[1])
    assert responses == {}


def test_missing_fallback_current_value_is_not_published(bitcoin, axes, responses):
    set_fallback(responses, bitcoin, name='Static', values=[None] * 12)

    with pytest.raises(ValueError, match=r'Static metric.*current'):
        asyncio.run(StaticMetric([math.nan] * 12).calculate(bitcoin, axes))


def test_notification_failure_does_not_prevent_fallback(
    bitcoin,
    axes,
    responses,
    monkeypatch,
):
    def unavailable_bot(_token):
        raise telegram.error.NetworkError('Telegram is unavailable')

    monkeypatch.setenv('TELEGRAM_TOKEN', 'test-token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', '123')
    monkeypatch.setattr(telegram, 'Bot', unavailable_bot)
    responses['https://woocharts.com/bitcoin-price-models/data/chart.json'] = (
        httpx.Response(503)
    )
    set_fallback(responses, bitcoin, name='Woobull', values=[0.42] + [None] * 11)

    values = asyncio.run(WoobullMetric().calculate(bitcoin, axes))

    assert values[-1] == 0.42
