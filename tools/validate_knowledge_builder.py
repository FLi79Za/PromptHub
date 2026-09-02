"""Run a disposable real-Ollama Knowledge Base Builder validation."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--generation-model", default="gemma3:4b")
    args = parser.parse_args()

    config_dir = Path(tempfile.mkdtemp(prefix="prompthub-knowledge-config-"))
    os.environ["PROMPTHUB_CONFIG_DIR"] = str(config_dir)

    import app as prompthub
    from ai_library import (
        AILibraryError,
        audit_collection,
        create_action,
        create_collection,
        create_resource,
        prepare_document,
        rebuild_collection,
        retrieve_knowledge,
        save_prepared_document,
    )

    with tempfile.TemporaryDirectory(prefix="prompthub-knowledge-e2e-") as temp:
        prompthub.DB_PATH = Path(temp) / "prompts.db"
        prompthub.TEMP_DIR = Path(temp) / "app-temp"
        prompthub.TEMP_DIR.mkdir()
        prompthub.app.config.update(TESTING=True)
        prompthub.init_db()
        before_prompts = 0

        with prompthub.get_db() as conn:
            collection_id = create_collection(
                conn, name="Disposable Knowledge Builder Validation", knowledge_domain="Synthetic camera guide",
                version_label="1.0", embedding_model=args.embedding_model, chunk_size=400, chunk_overlap=40,
            )
            collection = conn.execute("SELECT * FROM ai_knowledge_collections WHERE id=?", (collection_id,)).fetchone()

        source = (
            "# Sculpture camera rule\n\n"
            "For any production prompt about a sculpture, the documented camera treatment is a slow clockwise "
            "orbital camera move. Preserve that exact direction and motion when refining the prompt.\n\n"
            "# Lighting\n\nUse a cool rim light behind the sculpture."
        ).encode()
        prepared = prepare_document(
            "camera_rule.md", source, collection, prompthub.ollama_embed, ingestion_mode="optimised",
            provenance={"source_title": "Synthetic Camera Guide", "source_file": "camera_rule.md", "topic": "Camera movement", "source_pages": "1", "version_label": "1.0"},
        )
        with prompthub.get_db() as conn:
            document_id = save_prepared_document(conn, collection_id, prepared)
            passages = retrieve_knowledge(conn, collection_id, "How should the camera move around a sculpture?", prompthub.ollama_embed)
            assert passages and "clockwise orbital" in passages[0]["content"]
            system_id = create_resource(conn, kind="system", name="Disposable grounded director", content="Use retrieved documentation conservatively and return only the final production prompt.")
            template_id = create_resource(conn, kind="template", name="Disposable grounded task", content="Refine CURRENT_PROMPT using the documented camera and lighting rules.")
            action_id = create_action(conn, {"name": "Disposable grounded action", "model": args.generation_model,
                "system_instruction_id": system_id, "prompt_template_id": template_id,
                "knowledge_collection_id": collection_id, "enabled": True, "allow_runtime_instruction": True})
            rebuilt_chunks = rebuild_collection(conn, collection_id, prompthub.ollama_embed)
            audit = audit_collection(conn, collection_id, prompthub.ollama_list_models())

        response = prompthub.app.test_client().post(
            f"/api/ai/actions/{action_id}/execute",
            json={"content": "A marble sculpture stands in a dark gallery.", "instruction": "Apply the documented camera treatment."},
        )
        payload = response.get_json()
        assert response.status_code == 200, payload
        generated = payload["content"]
        assert "orbital" in generated.lower() and "clockwise" in generated.lower(), generated
        assert payload["sources"] and payload["sources"][0]["filename"] == "camera_rule.md"

        try:
            prepare_document("empty.md", b"", collection, prompthub.ollama_embed)
        except AILibraryError as exc:
            assert "no extractable text" in str(exc).lower()
        else:
            raise AssertionError("Empty document was accepted")

        with prompthub.get_db() as conn:
            conn.execute("DELETE FROM ai_knowledge_documents WHERE id=?", (document_id,))
            conn.execute("DELETE FROM ai_knowledge_collections WHERE id=?", (collection_id,))
            assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == before_prompts
            assert conn.execute("SELECT COUNT(*) FROM ai_knowledge_collections").fetchone()[0] == 0
            assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        print(f"embedding_model={args.embedding_model}")
        print(f"generation_model={args.generation_model}")
        print(f"retrieval_top={passages[0]['filename']} score={passages[0]['score']:.4f}")
        print(f"rebuilt_chunks={rebuilt_chunks}")
        print(f"audit_findings={audit['finding_count']}")
        print(f"generation={generated}")
        print("database_integrity=ok")
        print("validation=passed")


if __name__ == "__main__":
    main()
