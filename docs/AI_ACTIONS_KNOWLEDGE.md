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

## Knowledge Base Builder

The Knowledge Manager and PromptHub Codex plugin share the authenticated Integration API rather than requiring database access. A collection has a stable UUID and revision, optional domain/version labels, an embedding model, chunk settings, source provenance, extraction/index state, and rebuild history.

### PDF, multi-file, and folder workflows

- In the UI, open **AI → Knowledge Manager**, create a collection, then select several supported files in one import. Browser security does not expose arbitrary folders, so choose the files from the folder together.
- With Codex, say “make this a PromptHub knowledge base for this subject” and supply a PDF/file set/folder. The skill inspects sources, previews creation/imports, uses the returned revision for each apply, tests retrieval, and audits the result.
- `.txt`, `.md`, `.markdown`, and text-extractable `.pdf` are supported. Scanned PDFs require OCR first. Files are limited to 10 MB each and API JSON limits also apply to base64 uploads.

### RAW, OPTIMISED, and MERGE

- **RAW** keeps clean source structure and indexes extracted text directly.
- **OPTIMISED** (recommended) asks Codex to create source-grounded retrieval Markdown before import: useful headings, coherent topic splits, nearby rules/examples, readable tables, and removed repeated PDF furniture. PromptHub does not invent or silently rewrite content.
- **MERGE** begins by inspecting the existing collection and previewing every source. Duplicate content and mismatched version labels are surfaced. Applying a version conflict requires an explicit decision.

Optimised Markdown is retained as the imported knowledge document in SQLite with the source filename/title/pages/topic/version/status provenance supplied by Codex. If operators also want standalone Markdown files, keep the reviewed files beside the original source set; PromptHub's canonical searchable copy remains local in `prompts.db` and can be previewed from Knowledge Manager.

### Updating, replacing, and rebuilding

Re-importing the same filename updates that source and replaces only its chunks. Preview updates through the plugin and use the exact collection revision; stale operations are rejected. Use a clearly versioned filename or separate collection when guidance is incompatible. Removal and collection deletion remain confirmed UI operations. **Rebuild document** or **Rebuild collection** regenerates chunks/vectors from stored extracted text and does not rewrite source text.

### Retrieval testing and health

Each collection includes a retrieval tester. Results show rank, filename, relevance score, chunk index/ID, provenance fields, and retrieved text. Use it to distinguish extraction/chunking/embedding/retrieval problems from generation-model problems.

The health audit reports failed/empty/unindexed sources, exact and near duplicates, mixed version labels, unusually small/large documents, and unavailable embedding models. Findings are recommendations only; PromptHub never deletes or replaces content from heuristics.

### AI Actions

A collection remains reusable across actions. Codex can suggest a conservative System Instruction, Prompt Template, and one coherent starter action, then preview the full bundle before creation. The texts should reflect the actual domain and retrieved documentation; generic examples are not hard-coded into collection logic.

### Troubleshooting

- **Embedding model missing:** install/select the intended Ollama embedding model, then rebuild.
- **No extractable PDF text:** OCR the PDF and import the OCR result or clean Markdown.
- **Version conflict:** create a separate version collection, explicitly label coexistence, or approve replacement after comparison.
- **Poor retrieval:** inspect extracted text and chunks, split noisy sources, adjust chunk size/overlap, rebuild, and retest before changing the generation model.
- **Stale revision:** inspect the collection again and do not automatically retry the write.

Natural-language examples:

- “Make this PDF a PromptHub knowledge base for the subject it documents.”
- “Create an optimised collection from all supported files in this folder.”
- “Merge these newer guides into my collection and flag obsolete or contradictory material.”
- “Test whether this collection knows how to answer this workflow question.”
- “Audit this collection and propose one grounded AI Action.”

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
