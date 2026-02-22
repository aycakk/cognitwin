from src.database.vector_store import add_memory, search_memory

# 1) Hafızaya 2 kayıt yaz
add_memory("[USER_NAME_MASKED] said the final exam is at 17:00.", {"source": "chat"})
add_memory("Assignment deadline is 2026-02-28 23:59 for DataMining.", {"source": "note"})

# 2) Soru sor, en ilgili kayıtları çek
hits = search_memory("When is the final exam?", k=3)
for h in hits:
    print("-", h["text"], "| meta:", h["meta"])
