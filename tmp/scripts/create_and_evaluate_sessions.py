#!/usr/bin/env python3
"""
Create MLflow experiment, log sessions, and evaluate with GenAI.

This script demonstrates the complete workflow:
1. Create an MLflow experiment
2. Log multi-turn conversation sessions to the experiment
3. Add traces from sessions to an evaluation dataset
4. Evaluate the dataset using mlflow.genai.evaluate (GenAI Evaluate)

The script uses both built-in multi-turn scorers and single-turn scorers.
"""

import os
import tempfile
import uuid
import mlflow
from mlflow.genai.datasets import create_dataset
from mlflow.genai.scorers import (
    ConversationCompleteness,
    UserFrustration,
    Safety,
    RelevanceToQuery,
)

# Enable multi-turn evaluation feature
os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

# Configure tracking URI (using SQLite for datasets support)
# Change this to "http://localhost:5000" if you have a dev server running
db_path = os.path.join(tempfile.gettempdir(), "mlflow_sessions_test.db")
TRACKING_URI = f"sqlite:///{db_path}"

# Experiment configuration
EXPERIMENT_NAME = f"Multi-Turn Session Evaluation Demo {uuid.uuid4()}"
SCRIPT_TAG = "create_and_evaluate_sessions"


def chat_turn(user_message, assistant_response, session_id, turn_number):
    """
    Simulate a single turn in a conversation with proper trace structure.

    This creates a trace where:
    - Input: the user's message only
    - Output: the assistant's response only
    - Metadata: session ID and turn number for grouping

    Args:
        user_message: The user's input message
        assistant_response: The assistant's response
        session_id: Session identifier to group conversation turns
        turn_number: Turn number in the conversation

    Returns:
        The assistant's response
    """
    with mlflow.start_span(name="chat_turn", span_type="CHAT_MODEL") as span:
        # Set user message as input (must be a dict for dataset compatibility)
        span.set_inputs({"message": user_message})

        # In a real scenario, this is where you'd call the LLM:
        # response = llm.chat(user_message)
        # For testing, we use the pre-defined response
        response = assistant_response

        # Set assistant response as output (must be a dict for dataset compatibility)
        span.set_outputs({"response": response})

        # Update trace metadata for session tracking
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id,
                "turn_number": str(turn_number),
                "script": SCRIPT_TAG,
            }
        )

    return response


def log_helpful_session(run_id):
    """
    Log a helpful conversation where the assistant answers questions.

    This session should score well on:
    - ConversationCompleteness: All questions answered
    - UserFrustration: No frustration indicators
    - Safety: Safe content
    - RelevanceToQuery: Relevant responses

    Returns:
        session_id: The session identifier
    """
    session_id = f"helpful_session_{run_id}"

    print("\n" + "=" * 70)
    print(f"Logging Session: Helpful Restaurant Recommendations")
    print("=" * 70)

    # Turn 1: Initial question about restaurants
    print("  Turn 1: User asks about Italian restaurants")
    chat_turn(
        user_message="Can you recommend a good Italian restaurant in downtown?",
        assistant_response="I'd recommend Bella Vista! It's a family-owned Italian restaurant on Main Street with excellent pasta and a cozy atmosphere. They're known for their homemade gnocchi.",
        session_id=session_id,
        turn_number=1,
    )

    # Turn 2: Follow-up about pricing
    print("  Turn 2: User asks about pricing")
    chat_turn(
        user_message="Is it expensive?",
        assistant_response="It's moderately priced. Main dishes range from $15-$25. They also offer a lunch special for $12 that includes an appetizer, entree, and drink.",
        session_id=session_id,
        turn_number=2,
    )

    # Turn 3: Question about reservations
    print("  Turn 3: User asks about reservations")
    chat_turn(
        user_message="Do I need a reservation?",
        assistant_response="Yes, I'd recommend making a reservation, especially for dinner. They get quite busy on weekends. You can book online through their website or call them at (555) 123-4567.",
        session_id=session_id,
        turn_number=3,
    )

    print(f"✓ Logged 3 turns for session: {session_id}")
    print("  Expected scores: High completeness, low frustration, safe, relevant")

    return session_id


def log_unhelpful_session(run_id):
    """
    Log an unhelpful conversation where the assistant avoids questions.

    This session should score poorly on:
    - ConversationCompleteness: Questions not properly answered
    - UserFrustration: User shows signs of frustration

    Returns:
        session_id: The session identifier
    """
    session_id = f"unhelpful_session_{run_id}"

    print("\n" + "=" * 70)
    print(f"Logging Session: Unhelpful Support Chat")
    print("=" * 70)

    # Turn 1: Question about refund
    print("  Turn 1: User asks about refund")
    chat_turn(
        user_message="I'd like to request a refund for my order. How do I do that?",
        assistant_response="Thank you for contacting us! We appreciate your business.",
        session_id=session_id,
        turn_number=1,
    )

    # Turn 2: User repeats question
    print("  Turn 2: User repeats the refund question")
    chat_turn(
        user_message="You didn't answer my question. How can I get a refund?",
        assistant_response="We have many satisfied customers who love our products!",
        session_id=session_id,
        turn_number=2,
    )

    # Turn 3: User expresses frustration
    print("  Turn 3: User expresses frustration")
    chat_turn(
        user_message="This is frustrating. I just need to know the refund process!",
        assistant_response="Is there anything else I can help you with today?",
        session_id=session_id,
        turn_number=3,
    )

    # Turn 4: User gives up
    print("  Turn 4: User gives up")
    chat_turn(
        user_message="Never mind, I'll call customer service directly.",
        assistant_response="Great! Have a wonderful day!",
        session_id=session_id,
        turn_number=4,
    )

    print(f"✓ Logged 4 turns for session: {session_id}")
    print("  Expected scores: Low completeness, high frustration")

    return session_id


def log_mixed_session(run_id):
    """
    Log a conversation with both helpful and unhelpful elements.

    This session should score moderately on most metrics.

    Returns:
        session_id: The session identifier
    """
    session_id = f"mixed_session_{run_id}"

    print("\n" + "=" * 70)
    print(f"Logging Session: Mixed Quality Support")
    print("=" * 70)

    # Turn 1: Good answer
    print("  Turn 1: User asks about shipping")
    chat_turn(
        user_message="How long does shipping take?",
        assistant_response="Standard shipping takes 5-7 business days. We also offer express 2-day shipping for an additional $10.",
        session_id=session_id,
        turn_number=1,
    )

    # Turn 2: Partial answer
    print("  Turn 2: User asks about tracking")
    chat_turn(
        user_message="Can I track my order?",
        assistant_response="Yes, tracking is available. Check your email.",
        session_id=session_id,
        turn_number=2,
    )

    # Turn 3: Needs clarification
    print("  Turn 3: User asks for more details")
    chat_turn(
        user_message="I haven't received a tracking email. Can you help?",
        assistant_response="Hmm, that's unusual. Have you checked your spam folder?",
        session_id=session_id,
        turn_number=3,
    )

    print(f"✓ Logged 3 turns for session: {session_id}")
    print("  Expected scores: Moderate on most metrics")

    return session_id


def create_and_log_sessions():
    """
    Create an experiment and log multiple conversation sessions.

    Returns:
        tuple: (experiment_id, run_id, list of session_ids, list of trace objects)
    """
    print("\n" + "=" * 80)
    print("STEP 1: CREATE EXPERIMENT AND LOG SESSIONS")
    print("=" * 80)

    # Set tracking URI
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"\n✓ Tracking URI: {TRACKING_URI}")

    # Create or get experiment
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print(f"✓ Experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

    # Start a single run for all sessions
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"✓ Started run: {run_id}")

        # Log three different conversation sessions
        session_ids = []
        session_ids.append(log_helpful_session(run_id))
        session_ids.append(log_unhelpful_session(run_id))
        session_ids.append(log_mixed_session(run_id))

        # Search for all traces in this run
        print("\n" + "=" * 70)
        print("Retrieving logged traces...")
        print("=" * 70)

        # Return DataFrame (default) for merge_records compatibility
        traces = mlflow.search_traces(
            experiment_ids=[experiment_id],
            filter_string=f'run_id = "{run_id}"',
            #include_spans=False,  # We only need metadata
        )

        print(f"✓ Retrieved {len(traces)} traces")

        # Verify traces have session metadata (DataFrame has trace_metadata column)
        if 'trace_metadata' in traces.columns:
            traces_with_sessions = traces[
                traces['trace_metadata'].apply(
                    lambda x: isinstance(x, dict) and x.get("mlflow.trace.session") in session_ids
                )
            ]
            print(f"✓ {len(traces_with_sessions)} traces have valid session metadata")

        # Display session summary
        print("\nSession Summary:")
        for session_id in session_ids:
            if 'trace_metadata' in traces.columns:
                session_count = traces[
                    traces['trace_metadata'].apply(
                        lambda x: isinstance(x, dict) and x.get("mlflow.trace.session") == session_id
                    )
                ].shape[0]
                print(f"  - {session_id}: {session_count} traces")

    return experiment_id, run_id, session_ids, traces


def create_evaluation_dataset(experiment_id, run_id, traces):
    """
    Create an evaluation dataset and add traces to it.

    Args:
        experiment_id: The experiment ID
        run_id: The run ID that contains the traces
        traces: List of trace objects to add to the dataset

    Returns:
        EvaluationDataset: The created dataset with traces
    """
    print("\n" + "=" * 80)
    print("STEP 2: CREATE EVALUATION DATASET")
    print("=" * 80)
    

    # Create dataset
    dataset_name = f"conversation_dataset_{run_id}"
    print(f"\nCreating dataset: {dataset_name}")

    dataset = create_dataset(name=dataset_name, experiment_id=experiment_id)
    print(f"✓ Dataset created: {dataset_name}")

    # Add traces to dataset
    print(f"\nAdding {len(traces)} traces to dataset...")
    dataset.merge_records(traces)
    print(f"✓ Added traces to dataset")

    # Verify dataset contents
    df = dataset.to_df()
    print(f"\nDataset info:")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Has 'trace' column: {'trace' in df.columns}")

    return dataset


def evaluate_dataset_with_genai(dataset, experiment_id):
    """
    Evaluate the dataset using mlflow.genai.evaluate (GenAI Evaluate).

    This uses both single-turn and multi-turn scorers:
    - Single-turn: Safety, RelevanceToQuery
    - Multi-turn: ConversationCompleteness, UserFrustration

    Args:
        dataset: The evaluation dataset to evaluate
        experiment_id: The experiment ID to log results to

    Returns:
        EvaluationResult: The evaluation results
    """
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATE WITH MLFLOW GENAI EVALUATE")
    print("=" * 80)

    # Initialize scorers
    print("\nInitializing scorers...")
    scorers = [
        # Multi-turn scorers (evaluate entire conversations)
        ConversationCompleteness(),
        UserFrustration(),
        # Single-turn scorers (evaluate individual turns)
        Safety(),
        RelevanceToQuery(),
    ]

    print("Scorers configured:")
    print("  Multi-turn scorers:")
    print("    - ConversationCompleteness: Checks if questions are answered")
    print("    - UserFrustration: Detects signs of user frustration")
    print("  Single-turn scorers:")
    print("    - Safety: Checks for unsafe content")
    print("    - RelevanceToQuery: Checks response relevance")

    # Set experiment for evaluation results
    mlflow.set_experiment(experiment_id=experiment_id)

    # Run evaluation
    print("\nRunning mlflow.genai.evaluate...")
    print("This will evaluate the dataset and log results to MLflow...")

    results = mlflow.genai.evaluate(data=dataset, scorers=scorers)

    print(f"✓ Evaluation complete!")
    print(f"✓ Results logged to run: {results.run_id}")

    return results


def display_results(results):
    """
    Display evaluation results in a readable format.

    Args:
        results: EvaluationResult object from mlflow.genai.evaluate
    """
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATION RESULTS")
    print("=" * 80)

    # Display aggregate metrics
    print("\n📊 Aggregate Metrics:")
    print("-" * 70)
    if results.metrics:
        for metric_name, metric_value in results.metrics.items():
            print(f"  {metric_name}: {metric_value}")
    else:
        print("  No aggregate metrics available")

    # Display detailed results table
    print("\n📋 Detailed Results:")
    print("-" * 70)

    if results.tables:
        # Get the main results table
        main_table_key = list(results.tables.keys())[0]
        results_df = results.tables[main_table_key]

        # Find scorer columns
        scorer_cols = [
            col
            for col in results_df.columns
            if any(
                keyword in col.lower()
                for keyword in [
                    "conversation",
                    "frustration",
                    "safety",
                    "relevance",
                ]
            )
        ]

        # Display relevant columns
        if scorer_cols:
            print(f"Showing {len(scorer_cols)} scorer result columns:")
            for col in scorer_cols:
                print(f"  - {col}")

            print("\nResults by trace:")
            display_cols = ["trace_id"] + scorer_cols
            available_cols = [col for col in display_cols if col in results_df.columns]

            if available_cols:
                print(results_df[available_cols].to_string(index=False))
            else:
                print("Note: Some columns not available in results")
        else:
            print("Available columns:", results_df.columns.tolist())
            print("\nFirst few rows:")
            print(results_df.head())
    else:
        print("  No result tables available")

    # Display session-level insights
    print("\n🔍 Session-Level Insights:")
    print("-" * 70)
    print(
        "Multi-turn scorers (ConversationCompleteness, UserFrustration) evaluate entire sessions."
    )
    print(
        "Single-turn scorers (Safety, RelevanceToQuery) evaluate individual turns within sessions."
    )
    print(
        f"\nView detailed results in MLflow UI: {mlflow.get_tracking_uri()}"
    )
    print(f"Run ID: {results.run_id}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("MLflow GenAI Evaluate: Multi-Turn Session Evaluation Demo")
    print("=" * 80)
    print("\nThis script demonstrates:")
    print("  1. Creating an MLflow experiment")
    print("  2. Logging multi-turn conversation sessions")
    print("  3. Creating an evaluation dataset from traces")
    print("  4. Evaluating with mlflow.genai.evaluate (GenAI Evaluate)")
    print("=" * 80)

    try:
        # Step 1: Create experiment and log sessions
        experiment_id, run_id, session_ids, traces = create_and_log_sessions()

        # Step 2: Create evaluation dataset
        dataset = create_evaluation_dataset(experiment_id, run_id, traces)

        # Step 3: Evaluate with GenAI
        results = evaluate_dataset_with_genai(dataset, experiment_id)

        # Step 4: Display results
        display_results(results)

        # Final summary
        print("\n" + "=" * 80)
        print("✅ COMPLETE!")
        print("=" * 80)
        print(f"\nExperiment ID: {experiment_id}")
        print(f"Run ID: {run_id}")
        print(f"Evaluation Run ID: {results.run_id}")
        print(f"Sessions evaluated: {len(session_ids)}")
        print(f"Total traces: {len(traces)}")
        print(
            f"\nView results in MLflow UI: {mlflow.get_tracking_uri()}"
        )
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
