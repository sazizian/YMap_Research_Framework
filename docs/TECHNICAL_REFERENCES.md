# Primary implementation references

These are implementation references, not evidence for any manuscript result.

- Psycopg prepared statements and `prepare_threshold`: https://www.psycopg.org/psycopg3/docs/advanced/prepare.html
- Psycopg cursor execution: https://www.psycopg.org/psycopg3/docs/api/cursors.html
- Psycopg batching/pipeline behavior: https://www.psycopg.org/psycopg3/docs/advanced/pipeline.html
- PostgreSQL constraint triggers: https://www.postgresql.org/docs/14/sql-createtrigger.html
- PostgreSQL keys, checks and foreign keys: https://www.postgresql.org/docs/14/sql-createtable.html
- Alloy project/API source (the bridge targets the modern `edu.mit.csail.sdg` package names): https://github.com/AlloyTools/org.alloytools.alloy
- Hugging Face base model APIs: https://huggingface.co/docs/transformers/main_classes/model
- Scikit-learn group splits: https://scikit-learn.org/stable/modules/cross_validation.html
- PyTorch tensor-only state dictionaries: https://pytorch.org/docs/stable/notes/serialization.html

The bundled `VALIDATION_STATUS.json` states which paths were actually executed. API documentation does not substitute for a live Alloy/PostgreSQL/checkpoint integration check.
