"""
Multi-turn evaluation bug bash script.

This script logs multi-turn conversation traces to experiment 554741152990759220.
It is idempotent - running it multiple times will delete old traces and create fresh ones.
"""

import os
import mlflow

# Enable multi-turn evaluation feature
os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

# Set tracking URI to localhost server
mlflow.set_tracking_uri("http://localhost:5000")

# Target experiment for bug bash
EXPERIMENT_ID = "222228112133735103"
SCRIPT_TAG = "bug_bash_knock_knock"


def chat_turn(user_message, assistant_response, session_id, turn_number):
    """
    Simulate a single turn in a conversation with proper trace structure.

    This creates a trace where:
    - Input: the user's message only
    - Output: the assistant's response only

    Args:
        user_message: The user's input message
        assistant_response: The assistant's response
        session_id: Session identifier to group conversation turns
        turn_number: Turn number in the conversation

    Returns:
        The assistant's response
    """
    with mlflow.start_span(name="chat_turn", span_type="CHAT_MODEL") as span:
        # Set ONLY the user message as input
        span.set_inputs(user_message)

        # In a real scenario, this is where you'd call the LLM:
        # response = llm.chat(user_message)
        # For testing, we use the pre-defined response
        response = assistant_response

        # Set ONLY the assistant response as output
        span.set_outputs(response)

        # Update trace metadata for session tracking
        mlflow.update_current_trace(
            metadata={
                "mlflow.trace.session": session_id,
                "turn_number": str(turn_number),
                "script": SCRIPT_TAG,
            }
        )

    return response


def cleanup_existing_traces(experiment_id):
    """Delete traces created by this script to make it idempotent."""
    print("Checking for existing traces to clean up...")

    try:
        print("  - Searching for traces...")
        # NOTE: include_spans=False prevents hanging caused by ThreadPoolExecutor
        all_traces = mlflow.search_traces(
            locations=[experiment_id],
            return_type="list",
            max_results=1000,
            include_spans=False  # We only need metadata for cleanup, not full spans
        )
        print(f"  - Search completed. Found {len(all_traces)} total traces")

        # Filter for traces created by this script
        traces = [
            trace for trace in all_traces
            if trace.info.trace_metadata.get("script") == SCRIPT_TAG
        ]
        print(f"  - Filtered to {len(traces)} traces with script='{SCRIPT_TAG}'")

        if traces:
            print(f"Found {len(traces)} existing traces from this script. Deleting...")
            # Use delete_traces (plural) with trace_ids parameter
            trace_ids = [trace.info.trace_id for trace in traces]
            print(f"  - Calling delete_traces with {len(trace_ids)} trace IDs...")
            mlflow.MlflowClient().delete_traces(
                experiment_id=experiment_id,
                trace_ids=trace_ids
            )
            print(f"✓ Deleted {len(traces)} traces")
        else:
            print("✓ No existing traces found from this script")
    except Exception as e:
        print(f"Note: Could not clean up traces: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing anyway...")


def log_complete_conversation():
    """Log a conversation where all questions are answered (should pass ConversationalCompleteness)."""

    print("\n" + "=" * 70)
    print("Session 2: Complete Conversation (should pass)")
    print("=" * 70)

    with mlflow.start_run() as run:
        print(f"\n✓ Started run: {run.info.run_id}")

        session_id = f"complete_conversation_{run.info.run_id}"

        # Turn 1: User asks about weather
        print("\nTurn 1: User asks about weather")
        chat_turn(
            user_message="What's the weather like today?",
            assistant_response="It's sunny and 75°F with clear skies. A beautiful day!",
            session_id=session_id,
            turn_number=1
        )

        # Turn 2: User asks about recommendations
        print("Turn 2: User asks for activity recommendations")
        chat_turn(
            user_message="Great! What outdoor activities would you recommend?",
            assistant_response="With this weather, I'd recommend going for a hike, having a picnic in the park, or going for a bike ride along the river trail.",
            session_id=session_id,
            turn_number=2
        )

        # Turn 3: User asks about specific location
        print("Turn 3: User asks about best hiking trail")
        chat_turn(
            user_message="Which hiking trail would be best for beginners?",
            assistant_response="The Meadow Loop Trail is perfect for beginners. It's a 2-mile easy trail with beautiful wildflowers and only 200 feet of elevation gain.",
            session_id=session_id,
            turn_number=3
        )

        # Turn 4: User asks about parking
        print("Turn 4: User asks about parking")
        chat_turn(
            user_message="Is there parking available at the trailhead?",
            assistant_response="Yes, there's a free parking lot with 30 spaces at the Meadow Loop trailhead. It fills up quickly on weekends, so I'd recommend arriving before 9 AM.",
            session_id=session_id,
            turn_number=4
        )

        print(f"\n✓ Logged 4 turns for session: {session_id}")
        print("  Expected: ConversationalCompleteness = 'complete'")
        print("  Reason: All user questions were directly answered")

        return run.info.run_id, session_id


def log_incomplete_conversation():
    """Log a conversation where questions are ignored (should fail ConversationalCompleteness)."""

    print("\n" + "=" * 70)
    print("Session 3: Incomplete Conversation (should fail)")
    print("=" * 70)

    with mlflow.start_run() as run:
        print(f"\n✓ Started run: {run.info.run_id}")

        session_id = f"incomplete_conversation_{run.info.run_id}"

        # Turn 1: User asks about pricing
        print("\nTurn 1: User asks about pricing")
        chat_turn(
            user_message="How much does your premium plan cost?",
            assistant_response="We offer several great features in our premium plan!",
            session_id=session_id,
            turn_number=1
        )

        # Turn 2: User asks again about pricing (still not answered)
        print("Turn 2: User repeats pricing question")
        chat_turn(
            user_message="That's nice, but what's the actual price?",
            assistant_response="Our premium plan includes unlimited storage, priority support, and advanced analytics.",
            session_id=session_id,
            turn_number=2
        )

        # Turn 3: User asks about refund policy
        print("Turn 3: User asks about refunds")
        chat_turn(
            user_message="Okay, what's your refund policy if I'm not satisfied?",
            assistant_response="You can check out our customer testimonials to see how satisfied our users are!",
            session_id=session_id,
            turn_number=3
        )

        # Turn 4: User expresses frustration
        print("Turn 4: User asks direct question")
        chat_turn(
            user_message="Can you just tell me: is there a free trial available?",
            assistant_response="I'm glad you're interested in our service! Let me tell you about our enterprise features...",
            session_id=session_id,
            turn_number=4
        )

        print(f"\n✓ Logged 4 turns for session: {session_id}")
        print("  Expected: ConversationalCompleteness = 'incomplete'")
        print("  Reason: Assistant avoided answering direct questions about pricing, refunds, and free trial")

        return run.info.run_id, session_id


def main():
    print("=" * 70)
    print("Multi-Turn Bug Bash - Three Test Sessions")
    print("=" * 70)

    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

    # Clean up old traces first
    cleanup_existing_traces(EXPERIMENT_ID)

    # Session 1: Knock knock joke (baseline test)
    print("\n" + "=" * 70)
    print("Session 1: Knock Knock Joke (baseline)")
    print("=" * 70)

    with mlflow.start_run() as run:
        print(f"\n✓ Started run: {run.info.run_id}")

        session_id = f"knock_knock_{run.info.run_id}"

        # Turn 1: Knock knock
        print("\nTurn 1: Knock knock")
        chat_turn(
            user_message="Knock knock",
            assistant_response="Who's there?",
            session_id=session_id,
            turn_number=1
        )

        # Turn 2: Boo
        print("Turn 2: Boo")
        chat_turn(
            user_message="Boo",
            assistant_response="Boo who?",
            session_id=session_id,
            turn_number=2
        )

        # Turn 3: Punchline
        print("Turn 3: Don't cry, it's just a joke!")
        chat_turn(
            user_message="Don't cry, it's just a joke!",
            assistant_response="Ha! Good one. 😄",
            session_id=session_id,
            turn_number=3
        )

        print(f"\n✓ Logged 3 turns for session: {session_id}")
        session1_id = session_id

    # Session 2: Complete conversation
    run2_id, session2_id = log_complete_conversation()

    # Session 3: Incomplete conversation
    run3_id, session3_id = log_incomplete_conversation()

    # Verify all traces were created
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)
    all_traces = mlflow.search_traces(
        locations=[EXPERIMENT_ID],
        return_type="list",
        max_results=100,
        include_spans=False
    )
    traces = [
        trace for trace in all_traces
        if trace.info.trace_metadata.get("script") == SCRIPT_TAG
    ]
    print(f"✓ Total traces created: {len(traces)} (11 expected: 3 + 4 + 4)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Experiment ID: {EXPERIMENT_ID}")
    print(f"\nSession 1 (Knock Knock): {session1_id}")
    print(f"Session 2 (Complete - should pass): {session2_id}")
    print(f"Session 3 (Incomplete - should fail): {session3_id}")
    print(f"\nScript tag: {SCRIPT_TAG}")
    print("=" * 70)

    print("\n✓ All sessions logged successfully!")
    print(f"\nTo evaluate ConversationalCompleteness:")
    print(f"  1. Navigate to experiment {EXPERIMENT_ID}")
    print(f"  2. Use the multi-turn evaluation feature")
    print(f"  3. Apply ConversationalCompleteness scorer")
    print(f"  4. Session 2 should show 'complete'")
    print(f"  5. Session 3 should show 'incomplete'")


if __name__ == "__main__":
    main()
