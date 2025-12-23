# llm-edu-digest

Automatically collects academic papers on using LLMs for education from arXiv and uploads them to Zotero.

## Overview

This repository automates the process of:
1. Searching arXiv for papers related to LLMs in education
2. Downloading PDFs temporarily
3. Uploading metadata and PDFs to your Zotero library via the Zotero Web API
4. Committing a human-readable digest back to the repository

The workflow runs daily via GitHub Actions and generates a digest showing what papers were found and uploaded.

## Setup

### Prerequisites

- A Zotero account with API access
- GitHub repository with Actions enabled

### Zotero API Setup

1. Go to https://www.zotero.org/settings/keys
2. Create a new API key with read/write access to your library
3. Note your User ID (found at https://www.zotero.org/settings/keys)

### GitHub Secrets Configuration

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- `ZOTERO_LIBRARY_ID`: Your Zotero user ID or group ID
- `ZOTERO_API_KEY`: Your Zotero API key
- `ZOTERO_LIBRARY_TYPE`: Either `user` (default) or `group`

### Configuration

Edit `config.yml` to customize the search parameters:

```yaml
search_query: "cat:cs.AI OR cat:cs.CL OR cat:cs.LG AND (ti:education OR ti:learning OR ti:teaching OR ti:LLM OR ti:large language model)"
max_results: 10
sort_by: "submittedDate"  # Options: submittedDate, lastUpdatedDate, relevance
sort_order: "descending"   # Options: descending, ascending
```

## Usage

### Automatic Execution

The workflow runs automatically every day at 6 AM UTC. You can check the results in:
- The `digest.md` file in the repository
- Your Zotero library

### Manual Execution

To run the workflow manually:
1. Go to the "Actions" tab in your GitHub repository
2. Select "arXiv to Zotero Digest"
3. Click "Run workflow"

### Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ZOTERO_LIBRARY_ID="your_library_id"
export ZOTERO_API_KEY="your_api_key"
export ZOTERO_LIBRARY_TYPE="user"

# Run the script
python arxiv_to_zotero.py
```

## Files

- `arxiv_to_zotero.py`: Main script that searches arXiv and uploads to Zotero
- `config.yml`: Configuration for search parameters
- `requirements.txt`: Python dependencies
- `.github/workflows/arxiv-digest.yml`: GitHub Actions workflow
- `digest.md`: Generated digest of papers (updated by workflow)

## How It Works

1. **Search**: The script searches arXiv using the query specified in `config.yml`
2. **Download**: PDFs are downloaded to a temporary directory
3. **Upload**: Metadata and PDFs are uploaded to Zotero using the Web API
4. **Digest**: A markdown digest is generated with details about all papers
5. **Commit**: The digest is committed back to the repository

## License

MIT
