__all__ = ['create_frontend_payload', 'calc_returns', 'create_tearsheet', 'get_config', 'FORMAT']

# Cell

import pyfolio
from pyfolio.timeseries import perf_stats, gen_drawdown_table
import pandas as pd
from path import Path
from pprint import pprint
import datetime
import logging
import re

import simplejson as json
from .load_data import DATA_DIR, F_PAYLOAD_DIR
from .historical_bt import simulate_pnl
from .pnl_sim import get_pnl_reports

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def create_frontend_payload(file_name, force=False):
    new_file_name = F_PAYLOAD_DIR / file_name.basename().replace("payload_", "f_payload_", 1)
    have_file = False
    if not force:
        try:
            # Check if we already have the file we want to create
            with open(new_file_name) as f:
                have_file = bool(json.load(f))
        except:
            pass
        if have_file:
            return
    with open(file_name) as f:
        pay_j = json.load(f)

    events = pd.DataFrame.from_dict(pay_j["events"])
    events = events.set_index(pd.to_datetime(events.index))
    events["t1"] = pd.to_datetime(events["t1"])
    config = get_config(pay_j, file_name)

    closes, clf_signals, alpha_signals = get_pnl_reports(
        events,
        pay_j["symbols"],
        config["binarize"],
        config["binarize_params"],
    )
    primary_signals, secondary_signals = (
        (alpha_signals, clf_signals)
        if alpha_signals is not None
        else (clf_signals, alpha_signals)
    )
    primary_rets, pay_j["primary"]["pnl"] = create_tearsheet(
        closes, primary_signals, new_file_name, "primary"
    )
    if pay_j["secondary"]:
        _, pay_j["secondary"]["pnl"] = create_tearsheet(
            closes, secondary_signals, new_file_name, "secondary", primary_rets
        )

    # Delete stuff we don't want in the frontend payload
    del pay_j["events"]
    del pay_j["symbols"]

    logging.info(f"Writing f_payload at {new_file_name}")
    with open(new_file_name, "w") as f:
        json.dump(pay_j, f, ignore_nan=True, default=datetime.datetime.isoformat)
    return new_file_name


def calc_returns(df):
    df = df.resample("1B").last()

    if str(df.index.tz) != "UTC":
        df.index = df.index.tz_localize(tz="UTC")

    return df.pct_change()


def create_tearsheet(close, signal, file_name, report_type, benchmark_rets=None):
    logging.info(f"Creating {report_type} tearsheet for {file_name}")
    # Map long/short to long/flat
    # signal = (signal + 1) / 2
    pos_size = 10000
    df, df_wo_costs, cost_stats = simulate_pnl(close, signal, pos_size)
    returns = calc_returns(df)
    returns.name = report_type.title()
    returns_wo_costs = calc_returns(df_wo_costs)
    returns_wo_costs.name = report_type.title()

    if report_type == "primary":
        long_all = pd.DataFrame(1, columns=signal.columns, index=signal.index)
        df_bench, _, _ = simulate_pnl(close, long_all, pos_size)
        benchmark_rets = calc_returns(df_bench)
        benchmark_rets.name = "Benchmark (long all)"


    fig = pyfolio.create_returns_tear_sheet(
        returns, benchmark_rets=benchmark_rets, return_fig=True
    )
    fig_file_name = file_name.replace(".json", f"_{report_type}.png")
    fig.savefig(fig_file_name, bbox_inches="tight", pad_inches=0)

    p_stats = perf_stats(returns)
    p_stats_wo_costs = perf_stats(returns_wo_costs)
    dd_table = gen_drawdown_table(returns, 5)

    signal = signal.resample("1B").last()
    # Just-in-case normalize to 1 for reporting
    signal = signal / signal.max().max()
    signal.plot()
    signal = signal.set_index(signal.index.map(lambda x: x.isoformat()))


    return (
        returns,
        {
            "fig_file_name": str(Path(fig_file_name).basename()),
            "p_stats": p_stats.to_dict(),
            "p_stats_wo_costs": p_stats_wo_costs.to_dict(),
            "dd_table": dd_table.to_dict(),
            "signal": signal.to_csv(),  # CSVs are a lot more space-efficient for this dense 1500*50 table
            "cost_stats": cost_stats,
        },
    )


def get_config(payload, fn):
    if "config" in payload:
        return payload["config"]

    # bridge to the old payload format
    if "fixed_horizon" in fn:
        return {
            "binarize": "fixed_horizon",
            "binarize_params": int(re.findall(r"fixed_horizon_(\d+)", fn)[0]),
        }
    return {}