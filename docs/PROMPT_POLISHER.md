# Prompt Polisher

Import `skill_packages/prompt-polisher` from the Skills page using its absolute path. The existing importer stages bundled packages before replacing their destination, so importing from the checkout itself preserves the source. An identical re-import is idempotent.

Prompt Polisher declares Create, Transform, Convert, Refine and Diagnose in SKILL.md. The runtime honours these declarations; explicit local operation overrides still take precedence and legacy packages retain description-based discovery.

Choose Prompt Polisher in Skills or the editor's Apply Skill dialog. Select REFINE to improve a rough prompt without changing its intent. Select CONVERT and target **Qwen (local LLM)** to adapt it for Qwen. Execution Model selects the installed Ollama model that writes the result; Target selects the recipient of the finished prompt. No target means portable wording, regardless of the execution model.

The result schema separates the copy-ready `prompt` from `clarifying_questions` and `rationale`. Both UIs and their Copy actions use `result_text`, which contains only the prompt. Consequential assumptions and placeholders stay visible in that prompt. Running Apply Skill does not save or replace the draft until you explicitly accept it.

Tests: `python -m unittest discover -s tests`.
Regression coverage includes in-place and wrapped-ZIP imports, operation discovery, adapter precedence, Qwen selection, portable REFINE, completed execution traces and copy-ready result extraction. Runtime tests stub the writer; practical UI acceptance requires an available Ollama model.
