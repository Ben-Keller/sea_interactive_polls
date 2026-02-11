# SEA Lesson Poll Generation Pipeline (Robust)

This package generates 1 interactive poll per lesson JSON, inserts it as a `template_id: "poll"` segment, and exports:
- `poll_registry.json` (dedup memory)
- `exports/polls_en_map.json` (lesson_id -> poll object)

## What changed vs the old notebook
- Grounded poll prompts using a **Lesson Card** (title + key concepts/takeaways/resources + short excerpts).
- Strict schema validation.
- Dedup gate using TF-IDF cosine similarity across question+options.
- Critic scoring + auto-retry.
- No silent "0 polls" runs: if lessons aren't found or model isn't configured, it errors loudly.

## Install (local)
Python 3.10+ recommended.
```bash
pip install -U openai pydantic scikit-learn numpy
```

## Configure model
### Option A: OpenAI API
```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"   # or your preferred
```

### Option B: Azure OpenAI
```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
export AZURE_OPENAI_DEPLOYMENT="<deployment_name>"
```

## Run
Open `polls_pipeline.ipynb` and set:
- `LESSONS_ROOT`: path to your `/en` folder that contains Module_* subfolders.
- `OUTPUT_ROOT`: where to write updated lesson JSONs and exports.

Or run as script:
```bash
python polls_pipeline.py --lessons_root /path/to/en --output_root /path/to/03_Outputs/polls --target_per_lesson 1
```

## Notes
- The script generates polls only for **lesson files** where filename matches `<module>.<chapter>.<lesson>.json` and `lesson >= 1`.
  It skips chapter/module intros/outros like `x.y.0.json` or `x.y.-1.json`.
- If a lesson already contains a `poll` segment, it is skipped unless `--overwrite_existing_polls` is set.

