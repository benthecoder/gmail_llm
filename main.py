import base64
import datetime
import json
import logging
import os
import os.path
from typing import Dict, List, Optional, Union

import emoji
import tiktoken
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from openai import OpenAI
from tenacity import retry, stop_after_attempt

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"
MODEL = "gpt-4-1106-preview"
TOKEN_LENGTH_LIMIT = 128000

EMAILS = [
    "dan@tldrnewsletter.com",
    "hello@blogboard.io",
    "noreply@architecturenotes.co",
    "datamachina@substack.com",
    "importai@substack.com",
    "suraj@pointer.io",
    "daily@chartr.co",
    "thebatch@deeplearning.ai",
    "lon@dataelixir.com",
    "deeplearningweekly@substack.com",
    "gradientflow@substack.com",
    # "hello@digest.producthunt.com",
    # "notboring@substack.com",
    # "therundownai@mail.beehiiv.com",
    # "homescreen@mail.launchhouse.com",
]


def get_current_date() -> str:
    """Returns the current date in YYYY/MM/DD format."""
    return datetime.datetime.now().strftime("%Y/%m/%d")


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
            "Extract every possible link from a large collection of email newsletters. Your goal is to compile a comprehensive list of links, "
            "regardless of content type or category. Prioritize quantity, aiming for at least 50 links. Do not filter out any links; include everything you find.\n\n"
            "### Categorization: \n"
            "Organize the links into categories based on the content of the link. While the focus is on quantity, try to categorize the links into relevant groups, such as 'Technology', 'Health', 'Business', etc. "
            "If a link does not clearly fit into a category, place it under 'Miscellaneous' or a similar general category.\n\n"
            "### JSON Output Format: \n"
            "Present the extracted links in a JSON structure. Each category should be a key in the JSON object, and its value should be a list of link objects. Each link object must contain a 'title' and a 'link'. Example format:\n"
            "{\n"
            '  "Technology": [\n'
            '    { "title": "Tech News Article", "link": "http://example-tech.com" },\n'
            "    ...additional links...\n"
            "  ],\n"
            "  ...other categories...\n"
            "}\n\n"
            "### Special Consideration: \n"
            "- Aim to include a wide variety of links, ensuring you reach or exceed the target of 50 links.\n"
            "- Do not omit any links. The goal is to capture every link present in the email content.\n\n"
            "### Guidance: \n"
            "Ensure that your extraction process is thorough and expansive. The input consists of email newsletter content. Your output should be a diverse and extensive collection of links, organized into categories and presented in a clear JSON format."
        ),
    }

    email_body = " ".join(email["body"] for email in emails_data)
    user_message: Dict[str, str] = {"role": "user", "content": email_body}

    # TODO this doesn't account for the system and user message
    tk_len = get_token_length(email_body, model=model)
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
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError:
        return "Invalid JSON format."

    markdown_output = []

    logging.info("Getting links in markdown...")
    for category, items in data.items():
        markdown_output.append(f"### {category}\n")
        for item in items:
            title = item.get("title", "No Title")
            link = item.get("link", "#")
            markdown_output.append(f"- [{title.lower()}]({link})\n")

    return "".join(markdown_output)


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

    total_links = 0
    for category, items in json.loads(summary).items():
        total_links += len(items)
    print(f"Total links extracted: {total_links}")

    summary_md = convert_to_markdown(summary)

    date = get_current_date()
    os.makedirs("summary", exist_ok=True)
    with open(f"summary/{date.replace('/', '-')}.md", "w") as f:
        f.write(summary_md)


if __name__ == "__main__":
    main()
