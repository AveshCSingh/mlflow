#!/usr/bin/env python3
"""
Evaluate conversation traces for ConversationCompleteness and UserFrustration.

This script retrieves traces from a specific MLflow experiment and evaluates
them using built-in multi-turn conversation scorers.
"""

import os
import sys
import mlflow
from mlflow.genai.scorers import ConversationCompleteness, UserFrustration

# Experiment ID to evaluate
EXPERIMENT_ID = "554741152990759220"

# Tracking URI configuration
# Set this to your MLflow server URL or use environment variable
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main():
    # Set tracking URI if provided
    if TRACKING_URI:
        print(f"Setting tracking URI to: {TRACKING_URI}")
        mlflow.set_tracking_uri(TRACKING_URI)
    else:
        current_uri = mlflow.get_tracking_uri()
        print(f"Using current tracking URI: {current_uri}")

        # Check if we need a remote server for this experiment
        if current_uri.startswith("file://"):
            print("\nWARNING: You appear to be using a local file store.")
            print("If this experiment is on a remote MLflow server, please set MLFLOW_TRACKING_URI:")
            print("  export MLFLOW_TRACKING_URI=http://localhost:5000")
            print("  or")
            print("  export MLFLOW_TRACKING_URI=databricks")
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Exiting. Please set MLFLOW_TRACKING_URI and try again.")
                sys.exit(0)
    print(f"Retrieving traces from experiment: {EXPERIMENT_ID}")

    # Search for traces in the experiment
    # Note: We'll get all traces and filter for session metadata after
    traces = mlflow.search_traces(
        locations=[EXPERIMENT_ID],
    )

    # Filter for traces with session metadata
    if 'trace_metadata' in traces.columns:
        traces_with_sessions = traces[traces['trace_metadata'].apply(
            lambda x: isinstance(x, dict) and 'mlflow.trace.session' in x
        )]
        if len(traces_with_sessions) > 0:
            traces = traces_with_sessions

    print(f"Found {len(traces)} traces with session metadata")

    if len(traces) == 0:
        print("\nNo traces found with session metadata.")
        print("Make sure traces have been logged with mlflow.trace.session metadata.")
        return

    # Display trace information
    print(f"\nTrace columns: {list(traces.columns)[:10]}...")  # Show first 10 columns
    print(f"Number of traces: {len(traces)}")

    # Check if traces have metadata attribute
    if hasattr(traces, 'metadata') or 'metadata' in traces.columns:
        print("\nSample trace metadata:")
        if len(traces) > 0:
            first_trace = traces.iloc[0]
            if hasattr(first_trace, 'metadata'):
                print(f"  {first_trace.metadata}")
            elif 'metadata' in traces.columns:
                print(f"  {first_trace['metadata']}")

    # Initialize scorers
    print("\nInitializing scorers...")
    completeness_scorer = ConversationCompleteness()
    frustration_scorer = UserFrustration()

    # =========================================================================
    # PART 1: Directly invoke scorers on traces
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: DIRECTLY INVOKING SCORERS")
    print("="*80)

    print("\nNote: ConversationCompleteness and UserFrustration are session-level scorers")
    print("They should be invoked on a session (list of traces) not individual traces.")

    # Group traces by session - session-level scorers need all traces from a single session
    print("\nGrouping traces by session...")

    # Convert trace IDs to actual Trace objects and group by session
    print("\nFetching Trace objects from trace IDs...")
    sessions = {}  # session_id -> list of Trace objects
    for trace_id in traces['trace_id'].tolist():
        trace_obj = mlflow.get_trace(trace_id)
        # Get session ID from trace metadata
        session_id = trace_obj.info.trace_metadata.get('mlflow.trace.session', 'unknown')
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(trace_obj)

    print(f"Successfully fetched {sum(len(v) for v in sessions.values())} Trace objects")
    print(f"Grouped into {len(sessions)} session(s):")
    for session_id, session_traces in sessions.items():
        print(f"  - Session '{session_id}': {len(session_traces)} traces")

    # Invoke scorers on each session separately
    print("\nInvoking scorers on each session:")
    for session_id, session_traces in sessions.items():
        print(f"\n{'='*70}")
        print(f"Session: {session_id}")
        print(f"{'='*70}")

        # Invoke completeness scorer
        print(f"\n  ConversationCompleteness.run(session=...):")
        completeness_result = completeness_scorer.run(session=session_traces)
        print(f"    Result: {completeness_result}")

        # Invoke frustration scorer
        print(f"\n  UserFrustration.run(session=...):")
        frustration_result = frustration_scorer.run(session=session_traces)
        print(f"    Result: {frustration_result}")

        # Also demonstrate calling as a callable
        print(f"\n  Alternative: completeness_scorer(session=...):")
        callable_result = completeness_scorer(session=session_traces)
        print(f"    Result: {callable_result}")

    # =========================================================================
    # PART 2: Use mlflow.genai.evaluate
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: USING MLFLOW.GENAI.EVALUATE")
    print("="*80)

    # Set the experiment for logging evaluation results
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

    # Run evaluation
    print("\nRunning evaluation with ConversationCompleteness and UserFrustration scorers...")
    results = mlflow.genai.evaluate(
        data=traces,
        scorers=[completeness_scorer, frustration_scorer],
    )

    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Show the results dataframe
    print("\nScorer Metrics:")
    print(results.metrics)

    print("\nAvailable result tables:")
    print(list(results.tables.keys()))

    print("\nDetailed Results:")
    # Get the main results table
    if results.tables:
        main_table_key = list(results.tables.keys())[0]
        results_table = results.tables[main_table_key]

        # Show relevant columns from the results
        result_cols = [col for col in results_table.columns
                       if 'conversation' in col.lower() or 'frustration' in col.lower()]
        if result_cols:
            print(f"\nShowing columns: {result_cols}")
            print(results_table[result_cols])
        else:
            print("\nShowing all columns:")
            print(results_table)

    print("\n" + "="*80)
    print(f"Evaluation complete! Results logged to experiment {EXPERIMENT_ID}")
    print(f"Run ID: {results.run_id if hasattr(results, 'run_id') else 'N/A'}")
    print("="*80)


if __name__ == "__main__":
    main()
