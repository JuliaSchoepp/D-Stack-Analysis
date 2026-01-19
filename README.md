# Analyse des Deutschlandstack-Konsultationsprozesses

## Beschreibung

Dieses Projekt analysiert die Feedback-Issues aus dem Deutschlandstack-Konsultationsprozess. So ist es möglich, einen Eindruck über das Feedback zum Deutschland-Stack zu gewinnen, ohne alle 500+ issues auf GitLab durchzuklicken.

**Update**: Da das Konsultationsverfahren in eine zweite Runde gegangen ist werden fortlaufend einmal pro Nacht die aktuellen Issues von OpenCode abgefragt und ergänzt.

## Features

- **Daten-Ingestion**: Lädt Issues aus GitLab und bereinigt sie.
- **Sentiment-Analyse**: Verwendet Google Cloud Natural Language API zur Analyse der Stimmung.
- **Automatische Klassifizierung**: Nutzt Gemini (Google Generative AI) zur Label-Zuweisung basierend auf vordefinierten Kategorien.
- **Interaktive Visualisierung**: Streamlit-App für das Filtern und Anzeigen von Issues, Metriken und Diagrammen.
- **Stichproben**: Zeigt zufällige Issue-Beispiele basierend auf Filtern.


## Verwendung

Starte die Streamlit-App:
```bash
streamlit run App.py
```

Die App erlaubt:
- Filtern nach Labels, Seiten und Sentiment-Bereich.
- Anzeige von zufälligen Issue-Stichproben.
- Übersicht mit Metriken und Diagrammen (zeitliche Verteilung, Label-Häufigkeiten, Formular vs. manuelle Issues).

## Projektstruktur

- `App.py`: Streamlit-Anwendung für die Visualisierung.
- `ingest.ipynb`: Jupyter-Notebook für Daten-Ingestion und -Verarbeitung.
- `utils.py`: Hilfsfunktionen für Datenbereinigung und API-Aufrufe.
- `data/`: Verzeichnis für Parquet-Dateien.
- `keywords_config.txt`: Liste der Labels für die Klassifizierung.
- `requirements.txt`: Python-Abhängigkeiten.

## Beitragen

Beiträge sind willkommen! Bitte erstelle ein Issue oder Pull Request.

## Lizenz

MIT

