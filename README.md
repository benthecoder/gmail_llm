# Gmail LLM

![cover](/images/cover.png)

Using GPT4 to extract interesting links from my daily newsletters

**Why?** I spend too much time opening and checking through all my newsletters subscriptions, would rather have all interesting links in one page.

Inspired by [isafulf/inbox_cleaner](https://github.com/isafulf/inbox_cleaner)

## Images

![cli](/images/cli.png)

![email sample](/images/email.png)

## Prerequisites

- Python 3.10 or higher
- Gmail account
- Google Cloud account with Gmail API enabled
- OpenAI API key

## Setup

1. Clone repo:

   ```
   git clone https://github.com/benthecoder/gmail_llm.git
   cd gmail_llm
   ```

2. Install packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up Google API credentials:

   - create a new OAuth 2.0 Client ID ([instructions](https://developers.google.com/workspace/guides/create-credentials))
   - Download JSON file and rename it to `credentials.json`.
   - Put `credentials.json` in the `gmail_llm` directory.

4. Set OpenAI API key:

   - Get an OpenAI API key [here](https://platform.openai.com/api-keys)
   - Rename` .env.local` to `.env` and set your key

   ```
   OPENAI_API_KEY=<YOUR_API_KEY>
   ```

## Usage

Run the script:

```
python main.py
```

see an example in [summary](summary/2023-12-04.md) folder

## TODO

- [ ] improve prompt to extract as many links as possible and have better categorization
- [ ] Experiment with different forms of output, maybe article form? Have it copy someone's writing style, publish it to substack, could be a newsletter automation with AI thing
- [ ] Check if text extracted from emails can be cleaned up, any redundant info fed into prompt?
- [ ] Run this on cloud/local and dump results in DB so I can read into my blog or in notion

## Links

- Docs for listing emails [users.messages.list](https://developers.google.com/gmail/api/reference/rest/v1/users.messages/list)
- Docs for querying [Searching for Messages](https://developers.google.com/gmail/api/guides/filtering)
- [Counting tokens with tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
