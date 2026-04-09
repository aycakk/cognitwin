"""src.gates — shared gate decision logic.

Each gate file exports a pure decision function that returns
the boolean outcome plus a machine-readable reason code.
Callers (pipeline.py, student_agent.py) wrap these with their
own localized message strings, so extraction does not change
any user-visible text.

Step 2 of the migration plan. Gates are added one at a time:
  Step 2.1  c5_role_permission   ← this commit
  Step 2.2  c4_hallucination
  Step 2.3  c6_anti_sycophancy
  Step 2.4+ ... (see migration plan)
"""
