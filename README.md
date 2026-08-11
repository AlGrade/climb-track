# ClimbTrack

ClimbTrack ist eine lokale Offline-Pipeline für möglichst genaues und zeitlich stabiles
2D-Skeleton-Tracking in Boulder- und Klettervideos. Das Programm findet den Kletterer, verfolgt
ihn durch das Video, berechnet 308 Körperpunkte mit Meta Sapiens2-1B und repariert vorsichtig
kurze Modellfehler über die Zeit.

Das Projekt ist derzeit ausschließlich für private Nutzung vorgesehen.

## Projektziel und Grenzen

Phase 1 beantwortet genau diese Frage:

> Wo befinden sich die Körperpunkte des Kletterers in jedem einzelnen Videoframe?

Die erzeugten Daten sind die Grundlage für spätere Geschwindigkeiten, Gelenkwinkel und
Bewegungsmetriken. Solche Metriken sind aber bewusst noch nicht Teil dieses Projekts.

Nicht enthalten sind:

- Geschwindigkeits-, Winkel- oder Bewegungsmetriken;
- 3D-Rekonstruktion und Weltkoordinaten;
- Griff- oder Hold-Erkennung;
- Web-UI, Server oder Datenbank;
- Echtzeit- oder Mobile-Betrieb.

Qualität hat Vorrang vor Geschwindigkeit. Die komplette Pose-Berechnung eines kurzen Videos kann
auf einem Mac mehrere Stunden dauern. Alle teuren Schritte sind deshalb gecacht und fortsetzbar.

## Aktueller Stand

Die Milestones 1 bis 5 sind implementiert, am Video `best go.mp4` ausgeführt und in Git gesichert.

| Milestone | Ergebnis | Warum das nötig ist |
|---|---|---|
| 1 – Fundament | Video-Ingest, Metadaten, Frames, Konfiguration und Cache | Alle späteren Schritte erhalten reproduzierbare Bilder und korrekte Zeitstempel. |
| 2 – Kletterer finden | YOLO11x, ByteTrack, Kletterer-Auswahl, stabiler Pose-Crop und Kontrollvideo | Sapiens soll nur die richtige Person und trotzdem alle ausgestreckten Gliedmaßen sehen. |
| 3 – Rohes Skeleton | Sapiens2-1B, 308 Keypoints, TTA, Resume und Roh-Overlay | Liefert unveränderte Modellmessungen als überprüfbare Ausgangsbasis. |
| 4 – Messen | Lokales Annotationstool, zehn schwierige Frames, Ground Truth, PCK und OKS | Qualität wird gemessen, nicht nur nach Gefühl beurteilt. |
| 5 – Verbessern | Confidence-Gating, Ausreißer-/Swap-Erkennung, Interpolation und One-Euro-Filter | Kurze Fehler werden repariert, ohne schnelle echte Kletterbewegungen glattzubügeln. |
| 6 – geplant | ViTPose-Vergleichsbackend und Backend-Benchmark | Prüft objektiv, ob ein zweites Modell auf unseren Videos besser ist. |

### Resultat am Referenzvideo

Das Referenzvideo hat 1.648 Frames, variable Framerate und eine Dauer von rund 27,5 Sekunden.
Zehn gezielt schwierige Frames mit 40 bewegungsrelevanten Punkten wurden manuell kontrolliert.
Davon waren 394 Punkte im Bild sichtbar und auswertbar.

| Messwert | Roh | Nach Refinement |
|---|---:|---:|
| Mittlerer Fehler | 8,00 px | 3,31 px |
| PCK@0.2 | 98,98 % | 100,00 % |
| OKS-ähnlicher Score | 0,9859 | 0,9938 |
| 95. Fehlerperzentil | 12,22 px | 11,50 px |
| Rechter-Hand-Fehler | 61,94 px | 18,37 px |

Körper, zusätzliche Gelenkpunkte und Füße blieben in diesem Ground-Truth-Set unverändert bei
100 % PCK. Die linke Hand wurde durch das Refinement im Mittel leicht von 7,30 auf 9,56 Pixel
schlechter, blieb aber bei 100 % PCK. Die starke Verbesserung der rechten Hand überwiegt deutlich.
Zehn Stressframes sind eine sinnvolle Leitplanke, aber noch kein Beweis für jedes denkbare Video.

## Gesamtarchitektur

```text
Video
  │
  ▼
00_ingest ──► Frames + echte Quellzeitstempel + Metadaten
  │
  ▼
10_detect ──► Personen-Boxen von YOLO11x
  │
  ▼
20_track ───► stabile Personen-IDs von ByteTrack
  │
  ▼
25_select ──► genau ein ausgewählter Kletterer + stabiler 3:4-Pose-Crop
  │
  ▼
30_pose ────► 308 rohe Sapiens2-Keypoints pro Frame
  │
  ├─────────► Roh-Overlay zur Sichtkontrolle
  │
  ▼
40_refine ──► reparierte und zeitlich stabilisierte Keypoints
  │
  ├─────────► Roh-vs.-Refined-Vergleichsvideo
  │
  ▼
Annotation / Evaluation ──► Ground Truth, PCK und OKS
```

Alle neuronalen Modelle laufen lokal auf dem konfigurierten Gerät. In der aktuellen
Standardkonfiguration ist das `mps`, also Apples GPU-Schnittstelle auf dem Mac. Das Video und die
berechneten Posen werden nicht an einen Server geschickt. Eine Internetverbindung wird nur für den
expliziten, einmaligen Modelldownload benötigt.

## Schnellstart

### 1. Projekt öffnen

```bash
cd /Users/alexandergradenegger/IdeaProjects/klettervideo-skeleton-tracking
```

### 2. Umgebung installieren

Voraussetzungen:

- macOS auf Apple Silicon; andere Plattformen funktionieren bei passenden Tools ebenfalls;
- Python 3.12, verwaltet durch `uv`;
- `ffmpeg` und `ffprobe`;
- ausreichend Speicherplatz für Frames, Videos und das 6,08-GB-Posemodell.

```bash
brew install ffmpeg
uv sync --locked
```

Falls noch kein Lockfile existiert, einmal `uv sync` ausführen und das erzeugte `uv.lock`
committen.

### 3. Modelle herunterladen

Die Downloads erfolgen absichtlich nie automatisch:

```bash
uv run climbtrack download-yolo --config configs/default.yaml
uv run climbtrack download-sapiens --config configs/default.yaml
```

Vorher die Lizenzhinweise in `NOTICE.md` prüfen. Sapiens2-1B ist das freigegebene primäre
Pose-Backend. Milestone 2 verwendet Ultralytics YOLO11x/ByteTrack; vor einer Nutzung außerhalb
dieses privaten Projekts müssen insbesondere die jeweiligen Modell- und AGPL-3.0-Bedingungen neu
geprüft werden.

### 4. Installation prüfen

```bash
uv run climbtrack preflight --config configs/default.yaml
```

Der Befehl prüft Gerät, ffmpeg/ffprobe sowie Größe und SHA-256 der beiden Modelle. Ein fehlendes
Modell oder nicht verfügbares Gerät führt zu einem klaren Fehler. Es gibt keinen stillen Wechsel
auf CPU oder ein anderes Modell.

### 5. Ein Video vollständig analysieren

Für das vorhandene Referenzvideo:

```bash
uv run climbtrack run-all \
  "/Users/alexandergradenegger/Desktop/drive-download-20260810T065611Z-1-001/best go.mp4" \
  --config configs/default.yaml
```

Für ein anderes Video muss nicht erneut entwickelt oder annotiert werden. Einfach denselben Befehl
mit dem neuen absoluten Videopfad ausführen:

```bash
uv run climbtrack run-all "/absoluter/pfad/neues-video.mp4" \
  --config configs/default.yaml
```

`run-all` arbeitet alle Stufen in Abhängigkeitsreihenfolge ab. Bereits vorhandene, gültige
Ergebnisse werden aus dem Cache geladen. Das abschließende Terminal-Output nennt den Pfad zum
Vergleichsvideo `raw_vs_refined.mp4`.

## Lange Läufe stoppen und fortsetzen

Ein langer Lauf kann mit `Ctrl+C` gestoppt werden. Danach exakt denselben Command erneut starten.
Die besonders teure Sapiens-Stufe speichert jeden abgeschlossenen Frame in einem versteckten
`.work-<cache-key>`-Ordner. Beim Neustart werden bereits fertige Frames geprüft und übersprungen.

Wichtig:

- Für normales Fortsetzen **kein** `--force` verwenden.
- Video, Config, Modell und Code müssen gleich bleiben, damit derselbe Cache-Key entsteht.
- `--force` baut eine Stufe bewusst neu und verschiebt das alte Ergebnis in ein wiederherstellbares
  Backup.
- Ein laufender Download verwendet ebenfalls eine partielle Datei und kann fortgesetzt werden.

## Einzelne Befehle

Jeder Schritt kann separat gestartet werden. Die nötigen Vorgänger werden automatisch aus dem
Cache geladen oder ausgeführt.

```bash
# Video prüfen und verlustfreie Frames erzeugen
uv run climbtrack ingest "/path/to/video.mp4" --config configs/default.yaml

# Personen erkennen
uv run climbtrack detect "/path/to/video.mp4" --config configs/default.yaml

# Personen über die Zeit verfolgen
uv run climbtrack track "/path/to/video.mp4" --config configs/default.yaml

# Kletterer auswählen
uv run climbtrack select "/path/to/video.mp4" --config configs/default.yaml

# Kontrollvideo mit Boxen und Track-IDs
uv run climbtrack render-tracks "/path/to/video.mp4" --config configs/default.yaml

# Rohe Sapiens-Pose berechnen
uv run climbtrack pose "/path/to/video.mp4" --config configs/default.yaml

# Rohes Skeleton-Video rendern
uv run climbtrack render-pose "/path/to/video.mp4" --config configs/default.yaml

# Zeitliches Refinement aus dem vorhandenen Pose-Cache berechnen
uv run climbtrack refine "/path/to/video.mp4" --config configs/default.yaml

# Links roh, rechts refined rendern
uv run climbtrack render-comparison "/path/to/video.mp4" --config configs/default.yaml

# Vorhandene vollständige Cache-Einträge anzeigen
uv run climbtrack cache-list --config configs/default.yaml
```

## Falls die falsche Person ausgewählt wird

Die automatische Auswahl bewertet Track-Länge, Kontinuität, vertikale Bewegung, allgemeine
Bewegung, Position und Bildfläche. Wenn die beiden besten Kandidaten zu ähnlich sind, bricht das
Programm absichtlich ab, anstatt zu raten.

Alle IDs in einem Kontrollvideo anzeigen:

```bash
uv run climbtrack render-tracks "/path/to/video.mp4" --review-all \
  --config configs/default.yaml
```

Danach eine ID explizit verwenden:

```bash
uv run climbtrack run-all "/path/to/video.mp4" --track-id 7 \
  --config configs/default.yaml
```

Alternativ in einer lokalen Desktop-Sitzung auf die gewünschte Box klicken:

```bash
uv run climbtrack run-all "/path/to/video.mp4" --click \
  --config configs/default.yaml
```

## Was in den Milestones passiert ist

### Milestone 1 – Projektfundament, Ingest und Cache

- Python-3.12-Projekt mit `uv`, `pyproject.toml` und gelockten Abhängigkeiten aufgebaut.
- Strikte YAML-Konfiguration implementiert. Unbekannte Optionen werden abgelehnt, damit Tippfehler
  nicht unbemerkt falsche Ergebnisse erzeugen.
- Video mit ffprobe analysiert: Auflösung, Rotation, HDR, variable Framerate und echte Zeitstempel.
- Jeden Quellframe als PNG extrahiert. PNG vermeidet zusätzliche JPEG-Artefakte vor den Modellen.
- Kanonisches Frame-Parquet und Provenance für Config, Tools, Python, Betriebssystem und Git erzeugt.
- Inhaltsadressierten Cache mit SHA-256, atomarem Publish, Manifesten und Checksum-Prüfung gebaut.
- Fehlgeschlagene Builds werden getrennt aufbewahrt und nie als gültiger Cache verwendet.

Nutzen für das Endprodukt: Jeder spätere Messpunkt lässt sich exakt einem Frame und dessen echtem
Zeitpunkt zuordnen. Ergebnisse sind reproduzierbar und lange Arbeit muss nicht wiederholt werden.

### Milestone 2 – Detection, Tracking und Kletterer-Auswahl

- YOLO11x erkennt alle Personen pro Frame, nicht nur die auffälligste.
- ByteTrack verbindet die Boxen über die Zeit und vergibt stabile Track-IDs.
- Überlappende Teilpersonen-Boxen werden confidence-bewusst unterdrückt.
- Ein nachvollziehbares Scoring wählt den wahrscheinlichen Kletterer.
- Manuelle Auswahl über `--track-id` oder `--click` ergänzt; bei Unsicherheit harter Stopp.
- Kontrollvideo mit Box, Confidence, Track-ID, Framezahl und Originalaudio gebaut.
- Pose-Crop auf das offizielle Sapiens-Seitenverhältnis 768×1024 beziehungsweise 3:4 gebracht.
- Crop mit Bewegungsumfeld, 1,35-fachem Padding, kurzer Lückeninterpolation und 15-Frame-Glättung
  stabilisiert. Das grüne Rechteck wurde bewusst etwas großzügiger gelassen, damit Hände und Füße
  bei weiten Zügen nicht aus dem Modellbild fallen.

Nutzen für das Endprodukt: Andere Personen verwirren die Pose-Schätzung nicht, und ausgestreckte
Gliedmaßen bleiben im Sapiens-Eingabebild.

### Milestone 3 – Rohe Sapiens2-Pose

- Gepinntes Meta-Modell `facebook/sapiens2-pose-1b` lokal installiert und per SHA-256 verifiziert.
- Offizielle Transformers-/Safetensors-Implementierung statt der problematischen alten
  MMCV-Toolchain verwendet.
- Sapiens läuft lokal über Torch/MPS in 1024×768 Eingabeauflösung.
- Pro Frame werden alle 308 Goliath/Sociopticon-Keypoints samt unveränderter Confidence gespeichert.
- Horizontal-Flip-TTA tauscht Links/Rechts-Indizes korrekt; Multi-Scale-TTA mittelt vier
  Vorwärtsläufe pro Frame (`1.0` und `1.125`, jeweils normal und gespiegelt).
- Rohe Pose wird als `pose_raw.parquet` gespeichert; es findet noch keine zeitliche Korrektur statt.
- Framegenaues Resume für die mehrstündige Modellinferenz implementiert.
- Skeleton-Overlay VFR-sicher gerendert. Ein früher Fehler, bei dem variable Framerate Frames
  verlieren konnte, wurde korrigiert; das Referenzvideo enthält im Output wieder 1.648/1.648 Frames.

Warum im Video nicht alle 308 Punkte sichtbar sind: 238 Punkte gehören zum sehr dichten Gesicht.
Sie werden standardmäßig nicht gezeichnet, damit das Overlay lesbar bleibt. Von den 70 übrigen
Körper-, Hand- und Fußpunkten erscheinen nur jene, deren Confidence die Zeichenschwelle erreicht.
Die Parquet-Datei enthält trotzdem alle 308 Werte pro Frame.

Nutzen für das Endprodukt: Wir besitzen hochauflösende, detaillierte Rohmessungen und können jede
spätere Korrektur objektiv mit dem unveränderten Modelloutput vergleichen.

### Milestone 4 – Ground Truth und Evaluation

- Ein leichtgewichtiges lokales Matplotlib-Annotationstool gebaut.
- Zehn Stressframes werden deterministisch gewählt: überwiegend niedrige Confidence und starke
  Bewegung, ergänzt um Timeline-Abdeckung.
- Pro Frame werden 40 bewegungsrelevante Punkte geprüft: 17 Körperpunkte, 6 Fußpunkte,
  7 zusätzliche Schulter-/Ellbogen-/Nackenpunkte und 10 Fingerspitzen.
- Punkte lassen sich ziehen oder als nicht sichtbar markieren. Jede Änderung wird sofort gespeichert
  und eine abgebrochene Sitzung wird später fortgesetzt.
- Auswertung nach Pixelabweichung, normalisiertem PCK@0.2 und OKS-ähnlichem Score, insgesamt und
  getrennt nach Körper, Zusatzpunkten, Füßen und Händen.

Annotation starten:

```bash
uv run climbtrack annotate "/path/to/video.mp4" --config configs/default.yaml
```

Bedienung:

- Falschen Punkt mit der Maus verschieben.
- Unsichtbaren Punkt rechtsklicken oder über **Unsichtbar** markieren.
- Mit **Bestätigen + weiter** den gesamten Frame freigeben.

Rohe Pose gegen Ground Truth auswerten:

```bash
uv run climbtrack evaluate annotations/<video-session>/ground_truth.json \
  --config configs/default.yaml
```

Nutzen für das Endprodukt: Filter werden anhand echter Fehler eingestellt, nicht nach Bauchgefühl.

### Milestone 5 – Konservatives zeitliches Refinement

Stage 40 verwendet nur das bereits berechnete `pose_raw.parquet`; Sapiens wird dafür nicht erneut
gestartet. Die Pipeline führt in dieser Reihenfolge aus:

1. Confidence-Gating markiert unzuverlässige Punkte explizit als fehlend.
2. Unplausible Segmentlängen erkennen Ausreißer an Armen und Beinen.
3. Temporale Konsistenz entscheidet, welcher Endpunkt eines unplausiblen Segments verdächtig ist.
4. Kurze Lücken bis maximal fünf Frames werden interpoliert; lange Lücken bleiben missing.
5. Plötzliche Links/Rechts-Vertauschungen symmetrischer Punkte können repariert werden.
6. Ein adaptiver One-Euro-Filter glättet standardmäßig nur die detaillierten Hände.

Körper und Füße werden absichtlich nicht pauschal geglättet. Damit bleiben Dynos, Swings und Stürze
echte schnelle Bewegungen. Die Problemstelle um Sekunde 26 im Referenzvideo wurde kontrolliert;
der Sturz bleibt erhalten, während der kurze Handfehler deutlich reduziert wird.

Refinement evaluieren:

```bash
uv run climbtrack evaluate-refined "/path/to/video.mp4" \
  annotations/<video-session>/ground_truth.json \
  --config configs/default.yaml
```

Vergleichsvideo erzeugen:

```bash
uv run climbtrack render-comparison "/path/to/video.mp4" \
  --config configs/default.yaml
```

`raw_vs_refined.mp4` zeigt links die rohe und rechts die verbesserte Pose.

Nutzen für das Endprodukt: Die Zeitreihe wird zuverlässiger für spätere Ableitungen, ohne echte
schnelle Bewegungen künstlich zu verlangsamen.

## Cache und Ausgabedateien

Der Cache liegt standardmäßig im Projektordner unter `cache/`. Jeder Unterordnername ist ein
deterministischer Hash aus Video, relevanter Konfiguration, Implementierung, Modellen und direkten
Vorgängerartefakten.

```text
cache/
├── 00_ingest/<key>/
│   ├── frames/000000000.png
│   ├── frames.parquet
│   ├── metadata.json
│   └── ffprobe.json
├── 10_detect/<key>/detections.parquet
├── 20_track/<key>/tracks.parquet
├── 25_select/<key>/
│   ├── candidates.json
│   ├── selection.json
│   └── pose_crops.parquet
├── 30_pose/<key>/
│   ├── pose_raw.parquet
│   ├── keypoints.json
│   └── summary.json
├── 40_refine/<key>/
│   ├── pose_refined.parquet
│   └── summary.json
├── 50_render_tracks/<key>/tracking_overlay.mp4
├── 50_render_pose/<key>/skeleton_raw_overlay.mp4
└── 50_render_compare/<key>/raw_vs_refined.mp4

annotations/<video-session>/
├── ground_truth.json
├── evaluation.json
└── evaluation_refined.json
```

Die großen PNG-Frames befinden sich also unter `cache/00_ingest/<key>/frames/`. Beim Referenzlauf
belegt der gesamte Cache derzeit ungefähr 13 GB, davon rund 12 GB für verlustfreie Eingabeframes.
Die Modelle benötigen zusätzlich ungefähr 5,8 GB. Cache- und Modellordner sind nicht für Git
vorgesehen.

Ändert sich nur das Rendering, bleiben Ingest, Detection, Tracking und Pose gültig. Ändert sich
dagegen eine relevante Pose-Einstellung oder ein Vorgängerartefakt, entsteht absichtlich ein neuer
Cache-Key. Vollständige Einträge enthalten ein Manifest und Checksums für alle Artefakte. Builds
werden erst nach erfolgreichem Abschluss per atomarem Verzeichnis-Rename veröffentlicht.

## Datenformat

Pose-Daten werden zeilenweise als Parquet gespeichert:

```text
frame_idx, timestamp, track_id, keypoint_name,
x, y, confidence, is_missing, is_interpolated, source_backend
```

Wichtige Regeln:

- Koordinaten liegen im Originalbild, nicht nur im Crop.
- `timestamp` stammt aus dem Quellvideo und wird nicht aus einer angenommenen FPS rekonstruiert.
- Fehlende Punkte verwenden echte Arrow-Nullwerte für `x`, `y` und `confidence`, niemals `(0, 0)`.
- Roh-Confidence bleibt erhalten.
- Registry, Gruppenzuordnung, Links/Rechts-Paare und Skeleton-Kanten sind versioniert.
- Ein anderes Backend kann später in dasselbe kanonische Schema gemappt werden.

## Konfiguration

Alle Parameter stehen in `configs/default.yaml`. Für Experimente die Datei kopieren und eine eigene
Config an `--config` übergeben, statt Werte im Python-Code zu ändern.

Wichtige Bereiche:

- `project`: Cache, Annotationen, Seed und Gerät (`mps`, `cpu`, `cuda`).
- `ingest`: ffmpeg/ffprobe, PNG, HDR-Policy und Cache-Prüfung.
- `detection`: YOLO-Auflösung und Schwellenwerte.
- `tracking`: ByteTrack-Matching und Track-Puffer.
- `selection`: Mindestqualität und gewichtete Kletterer-Auswahl.
- `pose_crop`: Seitenverhältnis, Padding und zeitliche Crop-Stabilisierung.
- `pose`: Batchgröße, Präzision, Flip- und Multi-Scale-TTA.
- `pose_render`: Sichtbarkeit von Gesicht, Crop, Punkten und Vergleichsbreite.
- `annotation`: Anzahl Stressframes sowie PCK-/OKS-Parameter.
- `refine`: Confidence-Schwellen, Lückengröße, One-Euro-, Segment- und Swap-Parameter.
- `models`: unveränderliche Modellrevisionen, Dateigrößen und SHA-256-Hashes.

HDR wird standardmäßig mit `hdr_policy: fail` abgelehnt. `tonemap` verlangt passende
ffmpeg-Filter; `clip` akzeptiert bewusst möglichen Highlight-Verlust. Keine verlustbehaftete Policy
wird stillschweigend gewählt.

## Modell- und Download-Policy

YOLO11x liegt unter `models/yolo11x.pt`. Das primäre Posemodell ist fest gepinnt:

```text
Repository: facebook/sapiens2-pose-1b
Revision:   f5fed8b97b99698d5eea1d14ff0855d0b4c3f000
Datei:      model.safetensors
Größe:      6,08 GB
```

Die offiziellen Keypoint-Metadaten sind separat auf Sapiens2-Commit
`7e5bae88456ac418ff0e58e74106c9fe192055d4` gepinnt. Sie werden als Daten gelesen; heruntergeladener
Python-Code wird nicht ausgeführt. Downloads landen zunächst in einer temporären Datei, werden
gegen Größe und SHA-256 geprüft und erst danach atomar veröffentlicht. Inference arbeitet
`local-only` und kontaktiert Hugging Face nicht.

## Projektstruktur

```text
configs/                    YAML-Konfiguration
src/climbtrack/
├── annotation/             Stressframe-Auswahl, Editor und Evaluation
├── backends/               YOLO11x, ByteTrack und Sapiens2
├── cache/                  Cache-Manifeste, atomare Speicherung und Abhängigkeiten
├── model_downloads/        explizite, geprüfte Modelldownloads
├── refinement/             One-Euro-Filter und zeitliche Reparaturlogik
├── rendering/              gemeinsame Pose- und VFR-Videodarstellung
├── schema/                 kanonische Parquet- und Keypoint-Schemas
├── selection/              Kletterer-Scoring, Klickauswahl und Pose-Crops
├── stages/                 klar getrennte Pipeline-Stufen
├── video/                  ffprobe und deterministisches Frame-Decoding
├── cli.py                  alle CLI-Commands
├── config.py               strikte Config-Modelle
└── provenance.py           reproduzierbare Herkunftsdaten
tests/
├── unit/                   Tests der reinen Logik
└── integration/            kleiner synthetischer Video-Ingest-Test
annotations/                kleine, versionierte Ground-Truth-Sets
cache/                      große, reproduzierbare Resultate; nicht in Git
models/                     große Modellgewichte; nicht in Git
```

Die Struktur trennt Modelladapter, Pipeline-Steuerung, Datenschemas, Rendering und reine
Refinement-Logik. Es gibt keine vorgezogenen Abstraktionsebenen für hypothetische Backends; die
Schnittstellen sind aber so angelegt, dass ViTPose in Milestone 6 ergänzt werden kann.

## Reproduzierbarkeit und Sicherheitsentscheidungen

- Seeds werden gesetzt, soweit die verwendeten Backends deterministisches Verhalten erlauben.
- Configs lehnen unbekannte Keys ab.
- Modell- und Ergebnisdateien werden per SHA-256 geprüft.
- Cache-Einträge enthalten Config, Tool-/Paketversionen, Betriebssystem und Git-Provenance.
- Source-Timestamps und VFR werden erhalten; fehlende oder nicht monotone Timestamps sind Fehler.
- ffmpeg dekodiert deterministisch single-threaded und berücksichtigt Rotationsmetadaten.
- Es gibt keine stillen Modell-, Device- oder HDR-Fallbacks.
- Rohdaten und Refinement bleiben getrennt, damit Verbesserungen jederzeit überprüfbar sind.

## Entwicklung und Qualitätssicherung

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Aktueller Stand: 51 Tests bestehen. Getestet werden unter anderem Hashing, Cache-Verhalten,
Zeitstempel, Schemas, Scoring, ByteTrack, Pose-Crops, Pose-Resume, Keypoint-Registry, VFR-Rendering,
Annotation/Evaluation, Confidence-Gating, Interpolation, Swap-/Ausreißerlogik und One-Euro-Filter.
Neuronale Vollinferenz wird wegen Laufzeit und Modellgröße nicht im automatischen Test ausgeführt.

## Bekannte Grenzen

- Sapiens2-1B mit vier TTA-Durchläufen pro Frame ist auf Apple Silicon sehr langsam.
- Extreme Verdeckung kann keine Nachbearbeitung zuverlässig rekonstruieren.
- Lange fehlende Abschnitte bleiben absichtlich missing.
- Gesichtspunkte werden gespeichert, aber standardmäßig nicht gerendert oder annotiert.
- Das Ground-Truth-Set besteht bisher aus einem Video und zehn Stressframes.
- Refinement verbessert die rechte Hand deutlich, die linke Hand im Mittel jedoch nicht.
- Noch gibt es keinen objektiven Vergleich mit ViTPose.
- Perspektive, Weitwinkel und Kamerabewegung werden noch nicht in Weltkoordinaten umgerechnet.

## Nächster Milestone

Milestone 6 ergänzt ViTPose-H beziehungsweise ViTPose++ als zweites Backend, mappt dessen Punkte in
das bestehende kanonische Schema und führt beide Modelle auf demselben Ground-Truth-Set aus. Erst
die Messergebnisse entscheiden, ob ViTPose einen praktischen Vorteil bringt. Bis dahin ist
Sapiens2-1B das einzige produktiv verwendete Posemodell.
