# LLM Education Literature Digest

**Automated weekly watcher for research on Large Language Models in University Didactics**

This repository automatically searches arXiv and high-quality education journals (via OpenAlex) for new papers about LLMs in education. It produces a curated Markdown digest and imports papers with PDF attachments into your Zotero library.

---

## 🎯 What This Does

Every week (or on-demand), this workflow:

1. **Searches arXiv** for preprints about LLMs and education
2. **Searches OpenAlex** for papers in a curated whitelist of high-quality journals focused on:
   - University didactics and pedagogy
   - Management education
   - Operations research / analytics education
3. **Scores and filters** results using relevance scoring (LLM + education keywords)
4. **Downloads PDFs** when available from open access sources
5. **Imports to Zotero** with full metadata and PDF attachments (without citation keys initially)
6. **Syncs citation keys** from Zotero/Better BibTeX back to the database
7. **Generates a digest** (`digest.md`) with summaries and links
8. **Exports references** (`references.json`) for papers with synced citation keys
9. **Prevents duplicates** via persistent state tracking

---

## 🚀 Quick Start Guide

### Step 1: Fork This Repository

1. Click the **Fork** button at the top right of this page
2. Wait for GitHub to create your fork

### Step 2: Configure Secrets

You must add three **GitHub Actions Secrets** to your forked repository:

1. Go to your fork's **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add each of these:

| Secret Name | Description | How to Get It |
|------------|-------------|---------------|
| `ZOTERO_LIBRARY_ID` | Your numeric Zotero user library ID | Go to https://www.zotero.org/settings/keys → Look for "Your userID for use in API calls is XXXXXX" |
| `ZOTERO_API_KEY` | Write-enabled Zotero API key | On the same page, click "Create new private key" → Check "Allow library access" and "Allow write access" → Save the key |
| `OPEN_ALEX_API_KEY` | OpenAlex API key (optional but recommended) | Sign up at https://openalex.org/api-keys (or leave empty, but you'll be rate-limited) |

⚠️ **IMPORTANT:** 
- Never commit these secrets to the repository
- The `ZOTERO_API_KEY` must have **write access** enabled
- Keep your API keys private

### Step 3: Enable GitHub Actions

1. Go to the **Actions** tab in your forked repository
2. Click **"I understand my workflows, go ahead and enable them"**

### Step 4: Run Manually (Optional)

To test the setup before waiting for the weekly schedule:

1. Go to **Actions** → **Weekly Literature Watch**
2. Click **Run workflow** → **Run workflow**
3. Wait 2-5 minutes for the workflow to complete

---

## 📋 How to Check Results

### View the Digest

After a successful run:
1. Go to the **Code** tab
2. Open `digest.md` to see the curated list of papers

### View Workflow Logs

1. Go to **Actions** tab
2. Click on the latest workflow run
3. Click on the **watch** job
4. Expand each step to see detailed logs

Common log sections:
- **Searching arXiv**: Shows query and number of results
- **Searching OpenAlex**: Shows each journal searched
- **Curating Papers**: Shows filtering and scoring
- **Processing Papers**: Shows PDF downloads and Zotero imports

### Check Your Zotero Library

1. Go to https://www.zotero.org/
2. Log in and open your library
3. Look for newly imported items with:
   - Full metadata (title, authors, abstract, DOI, journal)
   - PDF attachments (when available)
   - Source tracking in the "Extra" field (e.g., "Work-ID: arxiv:2401.12345")

### Understanding Citation Keys

**This workflow uses Zotero + Better BibTeX as the source of truth for citation keys:**

1. **On Import**: Papers are imported to Zotero **without** citation keys initially
2. **Better BibTeX Generates Keys**: Open Zotero Desktop (with Better BibTeX plugin installed) to auto-generate citation keys in the format `{author}{year}{a/b/c...}`
3. **Sync Back**: On the next workflow run, citation keys are synced from Zotero back to the SQLite database
4. **References Export**: Only papers with synced citation keys appear in `references.json` (for stable Quarto/Pandoc citations)

**To enable citation keys:**
- Install [Better BibTeX for Zotero](https://retorque.re/zotero-better-bibtex/)
- Configure citation key format: `[auth][year]` (author + year, with auto-disambiguation)
- Open Zotero Desktop periodically to sync and generate keys

**Files generated:**
- `references.json` - Only items with synced citation keys (ready for citations)
- `references_pending.json` - Items still waiting for citation keys (for transparency)

### Download Artifacts

Every run uploads `digest.md` and `state.json` as artifacts:
1. Go to **Actions** → click on a workflow run
2. Scroll to **Artifacts** section at the bottom
3. Download **literature-watch-results.zip**

---

## ⚙️ Configuration

### Journal Whitelist

The file `journals.json` contains the curated list of journals to search. It includes:

**Tier A (Core university didactics):**
- Computers & Education
- The Internet and Higher Education
- Assessment & Evaluation in Higher Education
- Active Learning in Higher Education
- Higher Education
- Studies in Higher Education
- Teaching in Higher Education
- Higher Education Research & Development
- International Journal for Academic Development

**Tier B (Discipline-anchored or practice venues):**
- Academy of Management Learning & Education
- INFORMS Transactions on Education
- INFORMS Journal on Applied Analytics (stricter filtering to avoid irrelevant analytics papers)

### Search Terms

The workflow looks for papers containing:
- **LLM terms**: large language model, LLM, ChatGPT, GPT, generative AI, GenAI
- **Education terms**: education, higher education, teaching, learning, tutoring, pedagogy, classroom, course, curriculum, assessment, feedback, student, training, instruction

### Schedule

By default, the workflow runs **every Monday at 06:15 UTC**. To change this:
1. Edit `.github/workflows/weekly_literature_watch.yml`
2. Modify the cron schedule (format: `minute hour day month weekday`)
3. Example: `'0 12 * * 3'` = Wednesdays at 12:00 UTC

---

## 🔧 Troubleshooting

### "Workflow failed" or no results

**Check the logs:**
1. Go to **Actions** → click on the failed run
2. Look for error messages in red

**Common issues:**

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `Zotero credentials not configured` | Missing secrets | Add `ZOTERO_LIBRARY_ID` and `ZOTERO_API_KEY` in Settings → Secrets |
| `403 Forbidden` from Zotero | API key lacks write permissions | Regenerate your Zotero API key with write access enabled |
| `401 Unauthorized` from Zotero | Wrong library ID or API key | Double-check your `ZOTERO_LIBRARY_ID` and `ZOTERO_API_KEY` |
| `429 Too Many Requests` from OpenAlex | Rate limiting | Add `OPEN_ALEX_API_KEY` secret or wait and retry |
| `Could not resolve source ID` | Journal not found in OpenAlex | This is logged but workflow continues; check if journal name/ISSN needs updating |
| `No changes to commit` | No new papers found | Expected behavior when no new relevant papers are published |
| `Permission denied` when pushing | Missing repo write permissions | Check workflow file has `permissions: contents: write` |

### PDFs not attaching

**Possible causes:**
- No open access PDF available for the paper
- PDF URL is broken or requires authentication
- Network timeout during download

**What happens:**
- The paper is still imported to Zotero with metadata
- `attachment_missing=true` is noted in state (future re-check capability)
- Check the "Extra" field in Zotero for notes

### Duplicate items in Zotero

The workflow tracks seen items in `state.json` to prevent duplicates. If you see duplicates:
1. Check that `state.json` is being committed after each run
2. Manually remove duplicates from Zotero
3. Add their IDs to `state.json` (seen_arxiv_ids, seen_openalex_ids, or seen_dois)

### Workflow not running on schedule

GitHub may disable scheduled workflows if:
- No repository activity for 60 days
- The repository is new and scheduled workflows haven't been approved

**Solution:**
- Run the workflow manually once to "wake it up"
- Make a commit every few months to keep the repo active

---

## 📊 Understanding Relevance Scoring

Papers are scored based on keyword matches:

| Match | Points |
|-------|--------|
| LLM term in title | +6 |
| LLM term in abstract | +3 |
| Education term in title | +4 |
| Education term in abstract | +2 |
| Bonus phrase in title (e.g., "tutoring", "assessment") | +2 |
| Penalty phrase (e.g., "compiler", "CUDA") | -6 |

**Minimum scores:**
- Tier A journals: 8 points (default)
- Tier B education journals: 7-8 points
- Tier B practice journals (Applied Analytics): 10 points + must include education intent terms

---

## 🔒 Security & Privacy

- **No credentials are committed** to the repository
- All secrets are stored in GitHub Actions Secrets (encrypted at rest)
- Secrets are injected as environment variables at runtime only
- The workflow runs in an isolated GitHub-hosted runner
- State file contains only paper IDs (no sensitive data)

---

## 🛠️ Advanced Usage

### Modify Search Parameters

Edit `watch_literature.py` to change:
- `ARXIV_MAX_RESULTS`: Number of arXiv papers to fetch (default 100)
- `OPENALEX_LOOKBACK_DAYS`: How many days back to search OpenAlex (default 10)
- `MAX_NEW_ITEMS_TOTAL`: Maximum papers per run (default 50)

### Add More Journals

Edit `journals.json` to add journals to the whitelist. Include:
- Journal name
- Tier (A or B)
- ISSN (if available)
- Any custom rules (e.g., stricter education intent gating)

### Run Locally

For development/testing:

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/llm-edu-digest.git
cd llm-edu-digest

# Install dependencies
pip install -r requirements.txt

# Set environment variables (use your real credentials)
export ZOTERO_LIBRARY_ID="your_library_id"
export ZOTERO_API_KEY="your_api_key"
export OPEN_ALEX_API_KEY="your_openalex_key"  # optional

# Run the script
python watch_literature.py
```

---

## 📝 Files in This Repository

| File | Purpose |
|------|---------|
| `watch_literature.py` | Main Python script that orchestrates everything |
| `requirements.txt` | Python dependencies |
| `journals.json` | Curated whitelist of journals to search |
| `state.json` | Persistent state for deduplication (auto-updated) |
| `digest.md` | Generated digest of new papers (auto-updated) |
| `literature.db` | SQLite database storing all papers and citation key status (auto-updated) |
| `references.json` | CSL-JSON export of papers with synced citation keys (auto-updated) |
| `references_pending.json` | Papers waiting for citation keys from Zotero (auto-updated) |
| `.github/workflows/weekly_literature_watch.yml` | GitHub Actions workflow definition |
| `.gitignore` | Files to exclude from git (temp files, caches) |

---

## 📚 Resources

- **arXiv API**: https://info.arxiv.org/help/api/index.html
- **OpenAlex API**: https://docs.openalex.org/
- **Zotero API**: https://www.zotero.org/support/dev/web_api/v3/start
- **GitHub Actions**: https://docs.github.com/en/actions

---

## 🤝 Contributing

Suggestions for improving the journal whitelist, search terms, or scoring logic are welcome! Please open an issue or pull request.

---

## 📄 License

This project is provided as-is for research and educational purposes.
