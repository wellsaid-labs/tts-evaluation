""" Streamlit application for querying, parsing, and visualizing Google Cloud Platform logs.

Usage:
    $ PYTHONPATH=. streamlit run run/utils/gcp/logs.py --runner.magicEnabled=false
"""
import json
import re
import subprocess
import time
import typing
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import lib

st.set_page_config(layout="wide")

OPTIONS: typing.Tuple[typing.Tuple[str, timedelta], ...] = (
    ("Last 90 Days", timedelta(days=90)),
    ("Last 31 Days", timedelta(days=31)),
    ("Last 7 Days", timedelta(days=7)),
    ("Last 24 hours", timedelta(hours=24)),
    ("Last 12 hours", timedelta(hours=12)),
    ("Last 6 hours", timedelta(hours=6)),
    ("Last 3 hours", timedelta(hours=3)),
    ("Last 60 minutes", timedelta(minutes=60)),
    ("Last 30 minutes", timedelta(minutes=30)),
)


class GCPLog(typing.TypedDict):
    """
    TODO: Add missing fields.
    """

    timestamp: str
    textPayload: str


def get_timestamps(logs: typing.List[GCPLog]) -> typing.List[datetime]:
    """Get the timestamp for each log."""
    timestamps = [r["timestamp"] for r in logs]
    return [datetime.fromisoformat(t[:-1][:26]) for t in timestamps]


def run_query(
    start: datetime, end: datetime, query: str, fraction: float = 1.0, limit: int = 2000
) -> typing.List[GCPLog]:
    """Get logs between `start` and `end` where the "textPayload" has the substring `query`.

    Args:
        ...
        fraction: The percentage of logs to sample, randomly.
        limit: The number of logs to get per query.
    """
    results: typing.List[GCPLog] = []
    bar = st.progress(0)
    info = st.empty()
    start_loop = time.time()
    while True:
        upperbound = min(get_timestamps(results)) if len(results) > 0 else end
        bar.progress((end - upperbound).total_seconds() / (end - start).total_seconds())
        elapsed = lib.utils.seconds_to_str(time.time() - start_loop)
        info.info(f"Got {end - upperbound} of logs. {elapsed} has elapsed since the first query.")

        template = (
            'resource.type=("k8s_cluster" OR "k8s_node" OR "k8s_pod" OR "k8s_container") AND '
            'resource.labels.cluster_name="text-to-speech" AND '
            'resource.labels.project_id="voice-service-255602" AND '
            'resource.labels.location="us-central1-a" AND '
            f'textPayload : "{query}" AND '
            f'timestamp<="{upperbound.isoformat()}Z" AND '
            f'timestamp>="{start.isoformat()}Z"'
            + (f" AND sample(insertId, {fraction})" if fraction < 1.0 else "")
        )

        # Learn more:
        # https://cloud.google.com/logging/docs/reference/tools/gcloud-logging#reading_log_entries
        command = ["gcloud", "logging", "read", template.strip(), "--format", "json"]
        command += ["--limit", str(limit)]
        with st.spinner("Running Query..."):
            output = subprocess.check_output(command)
        length = len(results)
        results.extend(json.loads(output))
        if len(results) == length:
            break

    st.info(f"Queried {len(results)} logs.")
    bar.progress(1.0)
    bar.empty()
    return results


def main():
    st.title("Google Cloud Platform (GCP) Log Parser")
    st.write("Query, parse, and visualize up to 90 days of GCP logs.")

    label = "Google Cloud Logging Search String"
    query = st.text_input(label, value=st.session_state.get("query", ""))
    st.session_state["query"] = query

    label = "Select Time Range"
    option: typing.Tuple[str, timedelta] = st.selectbox(
        label,
        OPTIONS,
        index=st.session_state.get("index", 3),
        format_func=lambda k: k[0],  # type: ignore
    )
    st.session_state["index"] = OPTIONS.index(option)
    end = datetime.utcnow()
    start = end - option[1]

    cols = st.beta_columns([1, 1])
    label = "Query Limit"
    query_limit = cols[0].number_input(label, value=st.session_state.get("query_limit", 100))
    st.session_state["query_limit"] = query_limit
    label = "Random Sample"
    fraction = st.session_state.get("fraction", 1.0)
    fraction = cols[1].number_input(label, min_value=0.0, value=fraction, max_value=1.0, step=0.01)
    st.session_state["fraction"] = fraction

    cols = st.beta_columns([1, 1])
    regex = cols[0].text_input("Text Payload Regex", value=st.session_state.get("regex", ""))
    st.session_state["regex"] = regex
    group = cols[1].number_input("Regex Group", min_value=0, value=st.session_state.get("group", 0))
    st.session_state["group"] = group

    if not st.button("Run Query"):
        st.stop()

    logs = run_query(start, end, query, fraction, query_limit)
    # TODO: Adjust for the timezone automatically instead of manually.
    timestamps = [t - timedelta(hours=14) for t in get_timestamps(logs)]
    pattern = re.compile(regex)
    # TODO: Support different visualizations, as needed.
    results = [pattern.search(r["textPayload"].strip()) for r in logs]
    try:
        matched = [float(r.group(group)) for r in results if r is not None]
        st.info(f"Found {len(matched)} logs.")
    except ValueError:
        st.warning("Unable to extract metric to visualize from logs.")
        st.stop()

    # Learn more: https://discuss.streamlit.io/t/date-at-x-axis-of-line-chart/853/4
    df = pd.DataFrame({"date": timestamps, "value": matched})
    st.line_chart(df.rename(columns={"date": "index"}).set_index("index"))


if __name__ == "__main__":
    main()
