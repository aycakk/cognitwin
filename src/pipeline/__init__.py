"""src.pipeline — orchestration layer.

Houses the REDO engine, routing logic, and (eventually) the
student and developer pipeline runners once they are extracted
from src/services/api/pipeline.py.

Migration plan step 3. Modules added incrementally:
  Step 3.1  redo.py       ← this commit
  Step 3.2  redo.py       (run_redo_loop — de-duplicate the loop body)
  Step 3.3  router.py     (_resolve_mode extraction)
"""
