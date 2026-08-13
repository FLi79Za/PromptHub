# AI Actions and Knowledge Library

PromptHub's AI area is an additive extension of the existing Flask, Jinja, SQLite, and
Ollama implementation. It does not alter existing prompt content during migration.

## First use

1. Start PromptHub and open **AI** in the top navigation.
2. Edit or duplicate the starter System Instruction and Prompt Template, or create your own.
3. Create an AI Action. Choose a generation model, instruction, template, and optional
   knowledge collection. Leaving the model blank uses the current/default Ollama model.
4. Open a library prompt, choose the action on **Use Prompt**, optionally override the model
   or add a one-off instruction, then select **Use AI**.
5. Review/edit the returned text. **Apply to Final Prompt** changes only the on-screen text.
   Use **Save as Variant** or the confirmed **Overwrite Stored Prompt** action separately.

The starter `General Prompt Refinement` action is created once when schema version 2 is
first applied. It is ordinary editable user data, not a special hard-coded execution path.

## Knowledge collections

1. In **AI → Knowledge Library**, create a collection.
2. Set an installed Ollama embedding model. `nomic-embed-text` is the initial suggestion,
   but it is not forced; change it to any compatible installed embedding model.
3. Import `.txt`, `.md`, `.markdown`, or `.pdf` files (10 MB maximum per file).
   Re-importing the same filename updates that document and replaces its chunks atomically.
4. PromptHub extracts text, creates overlapping character chunks, requests an embedding for
   each chunk from Ollama, and stores the source text/chunks/vectors locally in `prompts.db`.
5. If the embedding model, chunk size, or overlap changes, use **Rebuild Index**.
6. Attach zero or one collection to an AI Action.

At run time, the action task, optional instruction, and current prompt form the retrieval
query. PromptHub embeds that query with the same model, ranks stored chunks with cosine
similarity, and sends the five highest-scoring passages. Reference passages, task text,
one-off instructions, and the current prompt use separate XML-style delimiters. Reference
content is explicitly treated as untrusted documentation in the system message.

PDF extraction uses the lightweight `pypdf` package listed in `requirements.txt`. If it is
not installed, TXT/Markdown continue to work and the UI reports a focused PDF dependency
message instead of failing the application.

## Storage and deletion behavior

Schema migration `2 / ai_actions_knowledge_library_v1` adds only these tables:

- `ai_resources`
- `ai_actions`
- `ai_knowledge_collections`
- `ai_knowledge_documents`
- `ai_knowledge_chunks`

All user-facing records use UUID identifiers. Deleting an instruction, template, or
collection sets the corresponding action reference to `NULL`; it does not delete the action
or any prompt. Deleting a collection cascades only to that collection's documents and chunks.

## Configuration and failures

- Generation uses the existing non-streaming `/api/generate` client.
- Embeddings use Ollama `/api/embed`.
- Set `OLLAMA_HOST` before launch to use a different compatible host. The default is
  `http://localhost:11434`.
- Missing services/models, timeouts, HTTP failures, empty generation responses, missing
  indexes, extraction errors, and embedding errors are shown in the UI/JSON response.
- An indexing failure is prepared before database mutation, so a half-indexed document is
  not saved.

## Verification

```powershell
$env:TEMP = '<writable-temp>'
$env:TMP = $env:TEMP
.\env\Scripts\python.exe -B -m unittest discover -s tests -v
```

Manual validation should include creating/editing/duplicating/deleting both resource types,
creating an action, importing a real document with a running embedding model, executing the
action from **Use Prompt**, cancelling one result, applying a second result, and confirming
that the stored prompt changes only after an explicit save action.

## Current limits

- Indexing and generation are synchronous HTTP operations; large imports keep the request
  open until Ollama finishes.
- Retrieval uses character-aware chunks rather than a model-specific tokenizer.
- The first version supports one knowledge collection per action and five retrieved passages.
- Scanned/image-only PDFs need OCR before import.
