import base64
import datetime
import json
import logging
import os
import os.path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Union

import emoji
import markdown
import pytz
import tiktoken
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from openai import OpenAI
from tenacity import retry, stop_after_attempt

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants and configurations

#
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"
EMAILS = ["dan@tldrnewsletter.com"]
EMAIL = "benthecoder07@gmail.com"

# model
MODEL = "gpt-4-1106-preview"
TOKEN_LENGTH_LIMIT = 128000

# configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# Utility functions
def get_current_date(
    specific_date: Optional[str] = None, timezone_str: str = "America/New_York"
) -> str:
    """
    Returns a date in YYYY/MM/DD format.
    If a specific date is provided, it returns that date.
    Otherwise, it returns the current date for a given timezone.

    Args:
        specific_date (str, optional): A specific date in YYYY/MM/DD format. Defaults to None.
        timezone_str (str, optional): Timezone for the current date. Defaults to "America/New_York".

    Returns:
        str: The formatted date string.
    """
    if specific_date:
        return specific_date

    tz = pytz.timezone(timezone_str)
    return datetime.datetime.now(tz).strftime("%Y/%m/%d")


def get_gmail_client() -> Resource:
    """Creates and returns a Gmail client."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def build_query(
    email_filter_list: Optional[List[str]],
    date_after_filter: Optional[str],
    date_before_filter: Optional[str],
) -> Optional[str]:
    """Builds the query string for fetching emails."""
    query_parts = []

    if email_filter_list:
        query_parts.append(" OR ".join(f"from:{email}" for email in email_filter_list))

    date_after = date_after_filter or get_current_date()
    query_parts.append(f"after:{date_after}")

    if date_before_filter:
        query_parts.append(f"before:{date_before_filter}")

    return " ".join(query_parts).strip() or None


def fetch_emails(
    gmail: Resource,
    email_filter_list: Optional[List[str]] = None,
    date_after_filter: Optional[str] = None,
    date_before_filter: Optional[str] = None,
) -> List[dict]:
    """Fetches emails based on the given filters."""
    query = build_query(email_filter_list, date_after_filter, date_before_filter)

    try:
        results = gmail.users().messages().list(userId="me", q=query).execute()
    except HttpError as error:
        raise RuntimeError(f"Failed to fetch emails: {error}")

    return results.get("messages", [])


def parse_email_body(msg: dict) -> str:
    parts = msg["payload"].get("parts", [])
    for part in parts:
        # only parse the first text/plain part, ignore the rest
        if part["mimeType"] == "text/plain":
            body = part["body"].get("data", "")
            body = base64.urlsafe_b64decode(body.encode("ASCII")).decode("utf-8")
            break
    else:
        body = ""

    return body


def parse_email_data(
    gmail: Resource, email_info: Dict[str, Union[str, List[str]]]
) -> Dict[str, Union[str, List[str]]]:
    try:
        msg = (
            gmail.users()
            .messages()
            .get(userId="me", id=email_info["id"], format="full")
            .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        return {}

    try:
        headers = msg["payload"]["headers"]
        subject = next(h["value"] for h in headers if h["name"] == "Subject")
        sender = next(h["value"] for h in headers if h["name"] == "From")
        body = parse_email_body(msg)
    except Exception as e:
        print(f"Failed to parse email: {e}")
        return {}

    email_data = {"subject": subject, "sender": sender, "body": body}

    return email_data


@retry(stop=stop_after_attempt(3))
def summarize_email(
    emails_data: List[Dict[str, Union[str, List[str]]]],
    model: str = MODEL,
    token_length: int = TOKEN_LENGTH_LIMIT,
) -> str:
    if not isinstance(emails_data, list):
        logger.error("Email data must be a list.")
        return ""

    system_message: Dict[str, str] = {
        "role": "system",
        "content": (
            "### Task: \n"
            "Review the content of a collection of email newsletters, which includes various articles or sections with corresponding URLs. Summarize each article or section concisely, akin to a Hacker News post title, and include the actual URL from the list provided at the end of the email.\n\n"
            "### Objective: \n"
            "Generate one-sentence summaries for each article or section that capture the essence of the content. Match each summary with its actual URL from the list provided at the end of the email.\n\n"
            "### Output Format: \n"
            "Produce the summaries in a structured format, with each summary paired with the actual URL. Organize the summaries under categories like 'Technology', 'Business', 'Design', 'Web Development', and 'General News'. Example JSON output:\n"
            "{\n"
            '  "Technology": [{"url": "<actual URL>", "summary": "<one-sentence summary>"}],\n'
            '  "Business": [{"url": "<actual URL>", "summary": "<one-sentence summary>"}],\n'
            "  ...other categories with summaries and corresponding URLs...\n"
            "}\n"
            "### Additional Instructions: \n"
            "- Keep summaries concise and limited to one sentence, similar in style to a Hacker News post title.\n"
            "- Accurately associate each summary with its corresponding actual URL from the list at the end of the email."
        ),
    }

    email_body = " ".join(email["body"] for email in emails_data)
    user_message: Dict[str, str] = {"role": "user", "content": email_body}

    combined_message = system_message["content"] + user_message["content"]
    tk_len = get_token_length(combined_message, model=model)
    print("Token length:", tk_len)
    if tk_len > token_length:
        logger.error("Token length {tk_len} exceeded for model: {model}.")
        return ""

    try:
        client = OpenAI()
        logging.info("Summarizing your email...")
        completion = client.chat.completions.create(
            model=model,
            messages=[system_message, user_message],
            temperature=0.9,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"Failed in summarizing your email with {model} : {e}")
        return ""

    try:
        json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        logger.error("Invalid JSON format.")
        return ""

    return completion.choices[0].message.content


def get_token_length(input: str, model="gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = len(encoding.encode(input))
    return num_tokens


def get_senders(email_data: dict):
    def filter_emoji(s):
        return emoji.replace_emoji(s, replace="")

    return {
        email["sender"].split("<")[0].strip(): filter_emoji(email["subject"]).strip()
        for email in email_data
    }


def convert_to_markdown(json_data: str) -> str:
    import json
    import logging

    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        return "Invalid JSON format."

    markdown_output = []

    logging.info("Converting summaries and links to markdown...")
    for category, items in data.items():
        markdown_output.append(f"### {category}\n")
        for item in items:
            summary = item.get("summary", "No Summary")
            link = item.get("url", "#")
            markdown_output.append(f"- [{summary}]({link})\n")

    return "".join(markdown_output)


def send_email(gmail, to, subject, body_md):
    """
    Send an email using the Gmail API.

    Args:
        gmail: The Gmail API client.
        to (str): The recipient of the email.
        subject (str): The subject of the email.
        body_md (str): The body of the email in markdown format.
    """
    # Convert markdown to HTML
    body_html = markdown.markdown(body_md)

    # Apply styling for fixed width, center alignment, and black link color
    styled_html = f"""
    <div style="max-width: 600px; margin: auto; text-align: left;">
        {body_html}
    </div>
    """

    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(styled_html, "html"))

    raw_message = base64.urlsafe_b64encode(message.as_string().encode("utf-8"))
    try:
        gmail.users().messages().send(
            userId="me", body={"raw": raw_message.decode("utf-8")}
        ).execute()
        print(f"Email sent to {to}")
    except HttpError as error:
        print(f"An error occurred: {error}")


def main():
    gm = get_gmail_client()
    mails = fetch_emails(gm, email_filter_list=EMAILS)
    email_data = [parse_email_data(gm, mail) for mail in mails]

    senders = get_senders(email_data)
    print("The following emails will be summarized:")
    for sender, subject in senders.items():
        print(f"- {sender}: {subject}")
    print()

    confirmation = input("Do you want to continue? (Y/n): ")
    if confirmation.lower() != "y":
        print("Aborting...")
        return

    summary = summarize_email(email_data)
    if not summary:
        print("No summary generated.")
        return

    output_choice = input(
        "Do you want to save the summary to a file, send it via email, or both? (file/email/both): "
    )
    date = get_current_date()
    summary_md = convert_to_markdown(summary)

    if output_choice.lower() in ["file", "both"]:
        os.makedirs("summary", exist_ok=True)
        file_path = f"summary/{date.replace('/', '-')}.md"
        with open(file_path, "w") as f:
            f.write(summary_md)
        print(f"Summary saved to {file_path}")

    if output_choice.lower() in ["email", "both"]:
        send_email(gm, EMAIL, f"Your newsletter summary for {date}", summary_md)
        print("Summary sent via email.")


if __name__ == "__main__":
    main()
