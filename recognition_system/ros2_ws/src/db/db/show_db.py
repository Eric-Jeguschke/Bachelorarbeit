import lancedb

def main():
    """
    Gibt den Inhalt aller DB-Tabellen (ohne Embedding-Spalten) in der Konsole aus
    Gibt bei LanceDB kein einfach aufzurufendes GUI für die DB-Tabellen, deswegen diese Funktion.
    """
    db = lancedb.connect("/db/storage")
    for table_name in ["profiles", "episodes", "episode_keywords", "memories"]:
        print(f"\n{table_name}:")
        try:
            db_table = db.open_table(table_name).to_pandas()
        except Exception:
            print("Tabelle existiert noch nicht.")
            continue
        if db_table.empty:
            print("Tabelle ist noch leer!!")
            continue
        # Embedding-Spalten ausblenden.
        columns = []
        for col in db_table.columns:
            if not col.endswith("_emb"):
                columns.append(col)
        print(db_table[columns].to_string(max_colwidth=20))

if __name__ == "__main__":
    main()
