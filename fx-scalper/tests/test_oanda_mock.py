"""OANDA wrappers with mocked ``oandapyV20`` — no network."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.oanda.account import AccountClient
from src.oanda.client import OandaClient
from src.oanda.instruments import InstrumentRegistry
from src.oanda.orders import OrderClient, OrderRequest
from src.utils.journal import Journal


def _make_mock_client(journal: Journal, responses: list[dict]) -> OandaClient:
    """Build an OandaClient whose underlying API returns queued responses."""
    api = MagicMock()
    api.request = MagicMock(side_effect=responses)
    return OandaClient(
        api=api,
        account_id="001-001-1234567-001",
        environment="practice",
        journal=journal,
    )


class TestAccountClient:
    def test_snapshot_parses(self, tmp_journal: Journal) -> None:
        client = _make_mock_client(
            tmp_journal,
            [
                {
                    "account": {
                        "balance": "500.00",
                        "NAV": "505.00",
                        "marginUsed": "10.00",
                        "marginAvailable": "495.00",
                        "unrealizedPL": "5.00",
                        "openTradeCount": 1,
                        "currency": "USD",
                    }
                }
            ],
        )
        snap = AccountClient(client).snapshot()
        assert snap.balance == 500.00
        assert snap.nav == 505.00
        assert snap.margin_used == 10.0
        assert snap.margin_available == 495.0
        assert snap.unrealized_pl == 5.0
        assert snap.open_trade_count == 1
        assert snap.currency == "USD"


class TestInstrumentRegistry:
    def test_load_and_lookup(self, tmp_journal: Journal) -> None:
        client = _make_mock_client(
            tmp_journal,
            [
                {
                    "instruments": [
                        {
                            "name": "EUR_USD",
                            "displayName": "EUR/USD",
                            "type": "CURRENCY",
                            "pipLocation": -4,
                            "displayPrecision": 5,
                            "tradeUnitsPrecision": 0,
                            "minimumTradeSize": "1",
                            "maximumTrailingStopDistance": "1.0",
                            "minimumTrailingStopDistance": "0.00005",
                            "marginRate": "0.02",
                        },
                        {
                            "name": "USD_JPY",
                            "displayName": "USD/JPY",
                            "type": "CURRENCY",
                            "pipLocation": -2,
                            "displayPrecision": 3,
                            "tradeUnitsPrecision": 0,
                            "minimumTradeSize": "1",
                            "maximumTrailingStopDistance": "100.0",
                            "minimumTrailingStopDistance": "0.005",
                            "marginRate": "0.02",
                        },
                    ]
                }
            ],
        )
        reg = InstrumentRegistry(client)
        reg.load(["EUR_USD", "USD_JPY"])
        eur = reg.get("EUR_USD")
        assert eur.pip_size == 1e-4
        assert eur.display_precision == 5
        jpy = reg.get("USD_JPY")
        assert abs(jpy.pip_size - 1e-2) < 1e-12

    def test_pip_value_usd_quote_usd(self, tmp_journal: Journal) -> None:
        client = _make_mock_client(
            tmp_journal,
            [
                {
                    "instruments": [
                        {
                            "name": "EUR_USD",
                            "pipLocation": -4,
                            "displayPrecision": 5,
                            "tradeUnitsPrecision": 0,
                            "minimumTradeSize": "1",
                            "maximumTrailingStopDistance": "1.0",
                            "minimumTrailingStopDistance": "0.00005",
                            "marginRate": "0.02",
                        }
                    ]
                }
            ],
        )
        reg = InstrumentRegistry(client)
        reg.load(["EUR_USD"])
        # 4629 units × 0.0001 = $0.4629 per pip (quote is USD).
        pv = reg.pip_value_usd("EUR_USD", 4629, current_price=1.08)
        assert abs(pv - 0.4629) < 1e-6


class TestOrderClient:
    def test_market_order_tags_magic_and_journals(self, tmp_journal: Journal) -> None:
        client = _make_mock_client(
            tmp_journal,
            [
                {"instruments": [_eur_usd_spec()]},
                {
                    "orderCreateTransaction": {"id": "123"},
                    "orderFillTransaction": {
                        "id": "124",
                        "price": "1.08001",
                        "tradeOpened": {"tradeID": "T1"},
                    },
                },
            ],
        )
        reg = InstrumentRegistry(client)
        reg.load(["EUR_USD"])
        orders = OrderClient(client, reg)

        req = OrderRequest(
            strategy="bb_rsi_mr",
            instrument="EUR_USD",
            side="LONG",
            units=4629,
            sl_price=1.0780,
            tp_price=1.0820,
        )
        result = orders.place_market_order(req)
        assert result.status == "FILLED"
        assert result.oanda_trade_id == "T1"

        # Inspect what got sent to OANDA — should carry our magic id.
        args, kwargs = client.api.request.call_args_list[-1]
        endpoint = args[0]
        data = endpoint.data
        assert "order" in data
        order = data["order"]
        assert order["type"] == "MARKET"
        assert order["instrument"] == "EUR_USD"
        assert order["units"] == "4629"
        assert "clientExtensions" in order
        ext = order["clientExtensions"]
        assert ext["id"].startswith("FXSCALPER-V1:bb_rsi_mr:")
        assert ext["tag"] == "bb_rsi_mr"
        assert "stopLossOnFill" in order
        assert "takeProfitOnFill" in order


def _eur_usd_spec() -> dict:
    return {
        "name": "EUR_USD",
        "pipLocation": -4,
        "displayPrecision": 5,
        "tradeUnitsPrecision": 0,
        "minimumTradeSize": "1",
        "maximumTrailingStopDistance": "1.0",
        "minimumTrailingStopDistance": "0.00005",
        "marginRate": "0.02",
    }
