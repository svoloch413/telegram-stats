import argparse
import json
import os
import time
import datetime
import collections
from openai import OpenAI
import matplotlib.pyplot as plt
import requests

# Initialize the OpenAI client using your API key.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Constants
BATCH_INFO_FILENAME = "batches_info.json"
BATCH_SIZE = 100  # Number of messages per batch

def load_batches_info():
    """Load saved batch info from disk if available."""
    if os.path.exists(BATCH_INFO_FILENAME):
        with open(BATCH_INFO_FILENAME, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {"batches": []}

def save_batches_info(info):
    """Save batch info to disk."""
    with open(BATCH_INFO_FILENAME, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

def build_prompt(message_text, target_object):
    """Build the prompt for sentiment analysis."""
    prompt = f"""You are a helpful NLP assistant. Given the following chat message, determine if the message mentions the concept "{target_object}".
Consider alternative names or synonyms (for example, if the concept is "Elon Musk", references like "Musk", "Elon", "Илон", or "маск" should count).

If the message mentions the concept, classify the sentiment expressed in the context as "positive", "negative", or "neutral".
If it does not, simply return that information.

Return your answer as valid JSON with two keys:
- "mention": a boolean indicating whether the concept is mentioned.
- "sentiment": if mention is true, a string ("positive", "negative", or "neutral"), otherwise null.

Message: "{message_text}" """
    return prompt

def get_message_text(msg):
    """Extract text from a message, joining parts if needed."""
    text_field = msg.get("text", "")
    if isinstance(text_field, list):
        text_parts = []
        for part in text_field:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return " ".join(text_parts)
    elif isinstance(text_field, str):
        return text_field
    return ""

def create_batch_file(chunk, start_index, target):
    """
    Create a JSONL file for a chunk of messages.
    Returns the filename and a mapping dictionary for that batch.
    """
    filename = f"batch_input_{start_index}.jsonl"
    mapping = {}
    with open(filename, "w", encoding="utf-8") as f:
        for i, msg in enumerate(chunk):
            custom_id = f"msg_{start_index + i}"
            text = get_message_text(msg).strip()
            prompt = build_prompt(text, target)
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": [
                        {"role": "system", "content": "You are a sentiment analysis assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 1000
                }
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            # Save metadata (date and participant) for this message.
            try:
                dt = datetime.datetime.fromisoformat(msg.get("date"))
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = "Unknown"
            participant = msg.get("from", "Unknown")
            mapping[custom_id] = {"date": date_str, "participant": participant}
    return filename, mapping

def upload_batch_file(file_path):
    """Upload the batch input file using the Files API."""
    with open(file_path, "rb") as f:
        file_resp = client.files.create(file=f, purpose="batch")
    print(f"Uploaded batch input file. File ID: {file_resp.id}")
    return file_resp.id

def create_batch(file_id):
    """Create a batch job with the given file ID."""
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Created batch with ID: {batch.id}")
    return batch

def poll_batch(batch_id, poll_interval=10):
    """Poll the batch job status until it is completed."""
    print(f"Polling batch {batch_id} status...")
    while True:
        batch_status = client.batches.retrieve(batch_id)
        status = batch_status.status
        print(f"Batch {batch_id} status: {status}")
        if status == "completed":
            return batch_status
        elif status in ("failed", "expired", "cancelled"):
            raise Exception(f"Batch {batch_id} processing failed with status: {status}")
        time.sleep(poll_interval)

def download_file_content(file_id):
    file_response = client.files.content(file_id)
    return file_response.text

def process_batch(batch_info):
    """
    For a given batch (from saved batch info), poll until completion,
    download the output, and process the results using the stored mapping.
    Returns aggregated time series, participant counts, and number of processed messages.
    """
    batch_id = batch_info["batch_id"]
    mapping = batch_info["mapping"]
    batch_status = poll_batch(batch_id, poll_interval=10)
    output_file_id = batch_status.output_file_id
    if not output_file_id:
        raise Exception(f"Batch {batch_id} completed but no output file found.")
    print(f"Batch {batch_id} completed. Output file ID: {output_file_id}")
    output_content = download_file_content(output_file_id)
    # Aggregate results for this batch.
    time_series = collections.defaultdict(lambda: {"positive": 0, "negative": 0})
    participant_counts = collections.defaultdict(lambda: {"positive": 0, "negative": 0})
    processed_count = 0 
    for line in output_content.strip().splitlines():
        try:
            result_line = json.loads(line)
        except Exception as e:
            print(f"Error parsing line in batch {batch_id}: {line}\nError: {e}")
            continue
        custom_id = result_line.get("custom_id")
        if not custom_id:
            continue
        if result_line.get("error") is not None:
            print(f"Request {custom_id} in batch {batch_id} failed with error: {result_line.get('error')}")
            continue
        response = result_line.get("response")
        if not response or response.get("status_code") != 200:
            print(f"Request {custom_id} in batch {batch_id} returned non-200 status.")
            continue
        body = response.get("body")
        if not body:
            continue
        choices = body.get("choices", [])
        if not choices:
            continue
        
        # Debugging: print raw response content.
        import re  # Ensure this is at the top of your file if not already imported

        content = choices[0].get("message", {}).get("content", "")
        print(f"DEBUG: Raw content for {custom_id} in batch {batch_id}: {repr(content)}")
        if not content or not content.strip():
            print(f"Warning: Empty or whitespace-only response content for {custom_id} in batch {batch_id}. Skipping.")
            continue

        # Remove markdown code block formatting if present.
        # This regex matches a string that starts with ``` (optionally followed by "json"), then any content, and ends with ``` on its own line.
        pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = match.group(1)
            print(f"DEBUG: Stripped markdown content for {custom_id}: {repr(content)}")

        try:
            analysis_result = json.loads(content)
        except Exception as e:
            print(f"Error parsing response content for {custom_id} in batch {batch_id}: {e}")
            continue


        if not analysis_result.get("mention", False):
            continue
        sentiment = analysis_result.get("sentiment", "").lower()
        if sentiment not in ["positive", "negative"]:
            continue
        meta = mapping.get(custom_id, {})
        date_str = meta.get("date", "Unknown")
        participant = meta.get("participant", "Unknown")
        time_series[date_str][sentiment] += 1
        participant_counts[participant][sentiment] += 1
        processed_count += 1
    print(f"Processed {processed_count} messages in batch {batch_id}.")
    return time_series, participant_counts, processed_count


def main():
    parser = argparse.ArgumentParser(
        description="Batch Chat Analysis Script with Recovery using OpenAI Batch API"
    )
    parser.add_argument("--json_file", type=str, default="result.json", help="Path to the JSON chat export")
    parser.add_argument("--target", type=str, required=True, help='Target object to track (e.g., "Elon Musk")')
    parser.add_argument("--process", type=int, default=0,
                        help="Number of hundreds of messages to process (e.g., 5 for 500 messages). If 0, process all messages.")
    args = parser.parse_args()

    # Load JSON data and filter eligible messages.
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", [])
    print(f"Loaded {len(messages)} messages.")

    eligible_messages = []
    for msg in messages:
        if msg.get("type") != "message":
            continue
        text = get_message_text(msg).strip()
        if not text:
            continue
        eligible_messages.append(msg)
    print(f"Found {len(eligible_messages)} eligible messages.")

    # Load existing batch info (if any) and determine how many messages have been processed.
    batches_info = load_batches_info()
    processed_batches = batches_info.get("batches", [])
    processed_count = sum(b["num_messages"] for b in processed_batches) if processed_batches else 0
    print(f"Already processed {processed_count} messages from previous batches.")

    # Determine how many messages to process overall.
    total_to_process = len(eligible_messages)
    if args.process > 0:
        limit = args.process * BATCH_SIZE
        total_to_process = min(limit, total_to_process)
        eligible_messages = eligible_messages[:total_to_process]
        print(f"Processing first {total_to_process} messages as specified by --process.")
    else:
        print("Processing all eligible messages.")

    # Create new batches for remaining messages.
    new_batches = []
    current_index = processed_count
    while current_index < len(eligible_messages):
        chunk = eligible_messages[current_index: current_index + BATCH_SIZE]
        if not chunk:
            break
        batch_filename, mapping = create_batch_file(chunk, current_index, args.target)
        print(f"Created batch input file '{batch_filename}' for messages {current_index} to {current_index + len(chunk) - 1}.")
        file_id = upload_batch_file(batch_filename)
        batch_obj = create_batch(file_id)
        batch_info = {
            "batch_id": batch_obj.id,
            "start_index": current_index,
            "num_messages": len(chunk),
            "mapping": mapping
        }
        new_batches.append(batch_info)
        current_index += len(chunk)

    # Append new batches to existing batch info and save.
    batches_info.setdefault("batches", []).extend(new_batches)
    save_batches_info(batches_info)
    total_batches = len(batches_info["batches"])
    print(f"Saved batches info with {total_batches} batches.")

    # Process results from all batches.
    overall_time_series = collections.defaultdict(lambda: {"positive": 0, "negative": 0})
    overall_participant_counts = collections.defaultdict(lambda: {"positive": 0, "negative": 0})
    overall_processed = 0

    for batch_info in batches_info["batches"]:
        ts, pc, count = process_batch(batch_info)
        overall_processed += count
        # Merge time-series data.
        for date, sentiments in ts.items():
            overall_time_series[date]["positive"] += sentiments.get("positive", 0)
            overall_time_series[date]["negative"] += sentiments.get("negative", 0)
        # Merge participant counts.
        for participant, sentiments in pc.items():
            overall_participant_counts[participant]["positive"] += sentiments.get("positive", 0)
            overall_participant_counts[participant]["negative"] += sentiments.get("negative", 0)

    print(f"Overall processed messages: {overall_processed}")

    # --- Visualization ---
    # Time-series plot.
    sorted_dates = sorted([d for d in overall_time_series if d != "Unknown"])
    dates = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in sorted_dates]
    positive_counts = [overall_time_series[d]["positive"] for d in sorted_dates]
    negative_counts = [overall_time_series[d]["negative"] for d in sorted_dates]

    if dates:
        plt.figure(figsize=(12, 6))
        plt.plot(dates, positive_counts, label="Positive Mentions", color="blue", marker="o")
        plt.plot(dates, negative_counts, label="Negative Mentions", color="red", marker="o")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title(f"Mention Sentiment Over Time for '{args.target}'")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Stacked bar chart per participant.
    participants = list(overall_participant_counts.keys())
    pos_counts_part = [overall_participant_counts[p]["positive"] for p in participants]
    neg_counts_part = [overall_participant_counts[p]["negative"] for p in participants]

    plt.figure(figsize=(12, 6))
    x = range(len(participants))
    plt.bar(x, pos_counts_part, color="blue", label="Positive")
    plt.bar(x, neg_counts_part, bottom=pos_counts_part, color="red", label="Negative")
    plt.xlabel("Participant")
    plt.ylabel("Count of Mentions")
    plt.title(f"Sentiment Breakdown per Participant for '{args.target}'")
    plt.xticks(x, participants, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

