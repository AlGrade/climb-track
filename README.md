# ClimbTrack

ClimbTrack ist eine lokale Offline-Pipeline für möglichst genaues und zeitlich stabiles
2D-Skeleton-Tracking in Boulder- und Klettervideos. Das Programm findet den Kletterer, verfolgt
ihn durch das Video, berechnet 308 Körperpunkte mit Meta Sapiens2-1B und repariert vorsichtig
kurze Modellfehler über die Zeit.

Das Projekt ist derzeit ausschließlich für private Nutzung vorgesehen.

## Phase 1: Projektziel und Grenzen

Phase 1 beantwortet genau diese Frage:

> Wo befinden sich die Körperpunkte des Kletterers in jedem einzelnen Videoframe?

Die erzeugten Daten sind die Grundlage für spätere Geschwindigkeiten, Gelenkwinkel und
Bewegungsmetriken. Solche Metriken sind aber bewusst noch nicht Teil dieses Projekts.

In Phase 1 waren bewusst nicht enthalten:

- Geschwindigkeits-, Winkel- oder Bewegungsmetriken;
- 3D-Rekonstruktion und Weltkoordinaten;
- Griff- oder Hold-Erkennung;
- Web-UI, Server oder Datenbank;
- Echtzeit- oder Mobile-Betrieb.

Phase 1 ist abgeschlossen. Phase 2 erweitert das Projekt jetzt um einen lokalen zugbasierten
Videoplayer sowie Geschwindigkeits- und Winkelmessungen. 3D, Griff-Erkennung, Kraftmessung,
Serverbetrieb und Echtzeit bleiben weiterhin außerhalb des Scopes.

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
| 6 – übersprungen | ViTPose-Vergleichsbackend | Wird vorerst nicht gebaut; Phase 2 verwendet die vorhandenen refined Sapiens-Daten. |

Phase 2 hat begonnen. **P2.1 – Zugdefinition, Datenformat und lokaler Zug-Player** ist
implementiert; die Grenzen wurden in **P2.2** am Referenzvideo kontrolliert und nachgeschärft.
**P2.3** berechnet und zeigt Hand- und Körpergeschwindigkeiten je Zug. Der Player wird aus dem
refined Skelett automatisch mit Zugkandidaten befüllt; manuelle Bearbeitung ist nur noch eine
optionale Kontrolle.

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

# Lokalen Phase-2-Zug-Player öffnen
uv run climbtrack player "/path/to/video.mp4" --config configs/default.yaml

# Nur die automatische Zugerkennung ausführen
uv run climbtrack detect-moves "/path/to/video.mp4" --config configs/default.yaml

# Hand- und Körpergeschwindigkeiten je Zug berechnen
uv run climbtrack measure-moves "/path/to/video.mp4" --config configs/default.yaml
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
├── 70_moves/<key>/
│   ├── moves_auto.parquet
│   └── summary.json
├── 80_move_metrics/<key>/
│   ├── move_metrics.parquet
│   ├── move_speed_timeline.parquet
│   ├── move_metrics.json
│   └── summary.json
├── 50_render_tracks/<key>/tracking_overlay.mp4
├── 50_render_pose/<key>/skeleton_raw_overlay.mp4
├── 50_render_compare/<key>/raw_vs_refined.mp4
└── 90_player_video/<key>/player_video.mp4

annotations/<video-session>/
├── ground_truth.json
├── evaluation.json
├── evaluation_refined.json
├── moves_ground_truth.json
└── moves.parquet
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
├── moves/                  automatische Zugerkennung und korrigierbare Ground Truth
├── player/                 lokaler Browser-Player, API und statische Oberfläche
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
Schnittstellen bleiben trotzdem erweiterbar. Der ursprünglich geplante ViTPose-Vergleich wird
bewusst übersprungen; für Phase 2 ist Sapiens2-1B die feste Datenquelle.

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

Aktueller Stand: 67 Tests bestehen. Getestet werden unter anderem Hashing, Cache-Verhalten,
Zeitstempel, Schemas, Scoring, ByteTrack, Pose-Crops, Pose-Resume, Keypoint-Registry, VFR-Rendering,
Annotation/Evaluation, Move-Schema, Video-Range-Streaming, revisionssicheres Speichern,
Confidence-Gating, Interpolation, Swap-/Ausreißerlogik und One-Euro-Filter. Neuronale Vollinferenz
wird wegen Laufzeit und Modellgröße nicht im automatischen Test ausgeführt.

## Bekannte Grenzen

- Sapiens2-1B mit vier TTA-Durchläufen pro Frame ist auf Apple Silicon sehr langsam.
- Extreme Verdeckung kann keine Nachbearbeitung zuverlässig rekonstruieren.
- Lange fehlende Abschnitte bleiben absichtlich missing.
- Gesichtspunkte werden gespeichert, aber standardmäßig nicht gerendert oder annotiert.
- Das Ground-Truth-Set besteht bisher aus einem Video und zehn Stressframes.
- Refinement verbessert die rechte Hand deutlich, die linke Hand im Mittel jedoch nicht.
- Der ursprünglich geplante objektive Vergleich mit ViTPose wird vorerst bewusst übersprungen.
- Perspektive, Weitwinkel und Kamerabewegung werden noch nicht in Weltkoordinaten umgerechnet.
- Der Handflächenpunkt springt, wenn sich die Zahl der sichtbaren Handanker ändert, weil der Median
  dann über eine andere Punktmenge gebildet wird. Am Referenzvideo betrifft das 21 von 1.648 Frames;
  dort liegt die Geschwindigkeit im Mittel etwa beim Ein- bis Zweieinhalbfachen der Umgebung. Das
  frühere breite Ableitungsfenster hat das verdeckt statt behoben. Keiner der drei berichteten
  Spitzenwerte liegt auf einem solchen Frame, aber ein Maximum in dieser Nähe ist mit Vorsicht zu
  lesen. Eine Mindestankerzahl oder confidence-gewichtete Handfläche wäre die eigentliche Lösung.

## Phase 2: Züge und Bewegungsmetriken

Phase 2 baut direkt auf `pose_refined.parquet` auf. Sapiens muss für die Entwicklung der
Zugerkennung, des Players und der Metriken nicht erneut über das Video laufen.

### Was genau ist ein Zug?

Für dieses Projekt gilt zunächst:

> Ein Zug ist ein Handzug von einer stabilen Handposition zu einer neuen stabilen Handposition.

Der Zug beginnt, wenn eine zuvor ruhige Hand ihre Position verlässt. Er endet, wenn dieselbe Hand
an einer neuen Position wieder für eine Mindestdauer ruhig bleibt. Das System klassifiziert den
Zug als `left`, `right` oder `both`. Bewegen sich beide Hände innerhalb eines kurzen gemeinsamen
Zeitfensters, wird das als ein beidarmiger Zug behandelt, zum Beispiel bei einem Dyno.

Diese Definition braucht einige Sicherheitsregeln:

- Eine kurze Korrektur auf demselben Griff darf nicht automatisch als neuer Zug zählen.
- Die Körperbewegung kann vor der Hand beginnen und nach dem Greifen weiterlaufen. Der Player zeigt
  deshalb einen kleinen Vor- und Nachlauf um die eigentliche Zuggrenze.
- Verdeckungen und falsche Handpunkte können automatische Grenzen verfälschen. Jede Erkennung erhält
  eine Confidence und bleibt im Player manuell korrigierbar.
- Ohne Griff-Erkennung kann das System nur aus der ruhigen Handposition auf einen Kontakt schließen.
  Es behauptet nicht, den tatsächlichen Griff sicher erkannt zu haben.
- „Bewegte Hand“ und „ziehende Hand“ sind nicht dasselbe. Die bewegte Hand greift zum nächsten
  Punkt; eine ruhige Stützhand kann gleichzeitig den Körper ziehen. Kräfte sind aus einem einzelnen
  2D-Video nicht messbar.

### Geplanter Datenfluss

```text
pose_refined.parquet
        │
        ▼
Hand- und Körper-Zeitreihen
        │
        ▼
automatische Zugkandidaten ──► moves.parquet
        │                           │
        │                           ├──► lokaler Zug-Player + manuelle Korrektur
        │                           │
        │                           └──► Geschwindigkeit pro Zug
        │
        └──────────────────────────────► Gelenkwinkel pro Zug
                                            │
                                            ▼
                                      move_metrics.parquet
```

Die originalen Videotimestamps bleiben die Zeitbasis. Geschwindigkeiten werden nicht aus einer
angenommenen konstanten FPS berechnet.

### Phase-2-Milestones

#### P2.1 – Zugdefinition, Datenformat und Zug-Player (implementiert)

Erstes sichtbares Ergebnis ist ein lokaler Player, der das Video mit eingezeichnetem Skelett zeigt
und zwei Betriebsarten anbietet:

- **Zugmodus:** genau einen Zug abspielen und am Ende automatisch pausieren;
- **Gesamtvideo:** das komplette Video normal abspielen; aktive Zugmarkierung und
  Geschwindigkeitskurve folgen dabei automatisch dem aktuellen Frame.

Die englische Player-Oberfläche bietet **Previous move**, **Play move**, **Next move** und **Full
video**. Sie ist bewusst auf Video, Zugliste und Messwerte reduziert. Frame-Schritte sowie die
Korrektur von Start, Ende und Handzuordnung bleiben unter **Edit boundaries** eingeklappt erreichbar;
Änderungen werden separat als Ground Truth gespeichert.

Beim Start lädt der Player `pose_refined.parquet`, bildet pro Hand einen robusten Handflächenpunkt
und erkennt Übergänge zwischen stabilen Positionen automatisch. Bereits fertige Pose- und
Refinement-Caches werden wiederverwendet; Sapiens läuft nicht erneut. Auf dem Referenzvideo erkennt
die aktuelle Konfiguration zwei abgeschlossene Handzüge und einen gescheiterten letzten Zug. Ein
erfolgreicher Zug endet nicht schon beim neuen Handkontakt, sondern erst, wenn sich auch Körper und
Beine beruhigt haben oder der nächste Zug beginnt. Bei einem Fehlversuch ersetzt der Sturz die sonst
erforderliche stabile Endposition: Der Abschnitt reicht von der Vorbereitung bis zum tiefsten Punkt
des Falls.

Für die sichtbare Wiedergabe verwendet der Player das bereits gecachte
`skeleton_raw_overlay.mp4`. Die Zuggrenzen selbst werden weiterhin aus den zeitlich geglätteten
Skelettdaten in `pose_refined.parquet` berechnet. Dadurch ist das Skelett im Player sichtbar, ohne
das Modell oder das Video erneut zu berechnen.

Die automatisch erkannten Abschnitte im Referenzvideo sind aktuell:

1. rechte Hand, `3,67–7,13 s`, abgeschlossen;
2. linke Hand, `7,13–12,09 s`, abgeschlossen;
3. linke Hand, `23,95–26,64 s`, Sturz.

Player starten:

```bash
uv run climbtrack player "/path/to/video.mp4" --config configs/default.yaml
```

Der Player öffnet auch dann, wenn die automatische Erkennung oder die Metrikstufe dieses Video
ablehnen. Beide Schritte sind für den Player nur Vorschläge; er ist zugleich der einzige Ort, an dem
Zuggrenzen korrigiert werden können. Ein harter Abbruch würde also genau das Werkzeug sperren, mit
dem der Fehler zu beheben wäre — etwa nachdem alle Züge gelöscht wurden oder wenn ein neues Video
keinen einzigen Zugkandidaten liefert. Das Terminal nennt den Grund deutlich, die Zugliste bleibt
leer und die Geschwindigkeitskurve blendet sich aus, bis die Grenzen von Hand gesetzt und der Player
neu gestartet wurde. Die eigenständigen Befehle `detect-moves` und `measure-moves` scheitern
weiterhin laut, weil dort das Ergebnis selbst das Ziel ist.

Der Befehl öffnet eine ausschließlich an `127.0.0.1` gebundene Browser-Oberfläche. Der Terminal
muss währenddessen geöffnet bleiben; `Ctrl+C` beendet den lokalen Player. Falls kein Browser
automatisch geöffnet werden soll, `--no-open-browser` verwenden und den ausgegebenen Link manuell
öffnen. Ist der konfigurierte Port belegt, probiert der Player automatisch den nächsten lokalen
Port; mit `--port 9000` kann ein Port erzwungen werden.

Beim ersten Start erzeugt Stufe `90_player_video` aus dem großen 4K-Skelettvideo eine
browserfreundliche 1080-Pixel-Version mit kurzen Schlüsselbildabständen. Das dauert nur die
Video-Umwandlung, verändert weder Analyse noch Originaldateien und wird danach aus dem Cache
wiederverwendet. Dadurch starten Wiedergabe und Frame-Sprünge insbesondere in Chrome deutlich
schneller.

Der linke Umschalter oben rechts im Video zeigt das aktuelle Layout und wechselt zwischen zwei
lokal gespeicherten Desktop-Ansichten. `Landscape layout` zeigt Video und Kurve kompakt untereinander.
`Portrait layout` gibt dem 9:16-Video ein passendes Fenster ohne breite schwarze Seitenflächen und
setzt die Zugsteuerung direkt darunter. Rechts stehen Zugliste und **Edit boundaries** nebeneinander;
Kurve und Messwerte folgen darunter über die volle Breite. Ein geöffneter Editor vergrößert diese
Zeile oder lässt die rechte Spalte intern scrollen, überlagert aber keine Kurve. Beide Varianten sind
auf einen Desktop-Viewport ohne Seiten-Scrollbar ausgelegt; auf schmalen Geräten bleibt normales
Scrollen als sichere responsive Darstellung erhalten.

Daneben liegt `Fullscreen`. Der Knopf schaltet gezielt nur das Videofenster in den Vollbildmodus
des Browsers. Kurve, Zugliste und Editor treten dabei bewusst zurück, weil im Vollbild ausschließlich
das Skelettvideo beurteilt werden soll. Die Bedienleiste zeigt dort zusätzlich die beiden
Frame-Knöpfe `−1` und `+1`. Damit funktioniert die framegenaue Navigation auch im Vollbild
vollständig per Mausklick; Zeitleiste, Wiedergabe und Ton bleiben ebenfalls bedienbar.
`f` schaltet den Vollbildmodus um, `Esc` verlässt ihn, und die Pfeiltasten springen weiterhin
einzelne Frames. Der Modus ist reine Darstellung: Er verändert weder Zuggrenzen noch Messwerte.

Während der Wiedergabe im Vollbild blenden sich Bedienleiste, Umschalter und Zugbeschriftung nach
gut zwei Sekunden Ruhe aus, damit nichts den Kletterer verdeckt; jede Maus- oder Tastenbewegung holt
sie sofort zurück. Bei **pausiertem** Video bleiben sie dauerhaft stehen, denn genau dann wird Frame
für Frame geprüft. Solange der Zeiger über der Leiste liegt oder ein Bedienelement den Fokus hat,
wird ebenfalls nicht ausgeblendet.

Der Player speichert jede Änderung sofort und atomar in:

```text
annotations/<video-session>/moves_ground_truth.json
annotations/<video-session>/moves.parquet
```

`cache/70_moves/<key>/moves_auto.parquet` enthält die unveränderten automatischen Kandidaten. JSON
im Annotation-Ordner ist die korrigierbare Sitzung; das dortige Parquet enthält denselben aktuellen
Stand für die spätere Metrikpipeline. Eine Revisionsnummer verhindert, dass zwei gleichzeitig
offene Browser-Tabs unbemerkt Änderungen überschreiben.

Kanonisches Zug-Schema:

```text
move_id, start_frame, end_frame, start_timestamp, end_timestamp,
moving_hand, confidence, source, is_reviewed, outcome
```

`outcome` unterscheidet `completed` (abgeschlossener Zug) von `fall` (gescheiterter Zug mit Sturz).

#### P2.2 – Automatische Zugerkennung evaluieren und tunen (Referenzvideo abgeschlossen)

Eine konservative erste automatische Erkennung ist bereits Teil von P2.1. Sie kombiniert:

- Handgeschwindigkeit aus den echten Quellzeitstempeln;
- stabile Phasen vor und nach der Bewegung;
- das Nachschwingen von Körper und Beinen nach dem Handkontakt;
- Mindestweg und Mindestdauer;
- eine robuste, aus mehreren Punkten gebildete Handflächenposition;
- körpergrößennormierte Schwellenwerte.

Ein terminaler Fehlversuch wird gesondert erkannt, weil nach ihm naturgemäß keine neue stabile
Handposition mehr existiert. Dazu kombiniert die Erkennung das Lösen der Hand mit der deutlichen
Abwärtsbewegung des Rumpfs und markiert den Zug als `fall`.

Als Handposition dient nicht eine einzelne Fingerspitze. Dafür wird ein robuster Handflächenpunkt
aus Handgelenk und mehreren verfügbaren Handpunkten gebildet. So erzeugt das natürliche Zittern
einzelner Finger nicht fälschlich einen neuen Zug.

Am Referenzvideo wurden Anzahl, Handseite und Zuggrenzen im Player kontrolliert. Dieses Feedback hat
die Startschwelle, das Ende nach der Körperberuhigung und die Sonderbehandlung eines terminalen
Sturzes festgelegt. Ergebnis sind zwei abgeschlossene Züge und ein als `fall` markierter Fehlzug.
Für weitere Videos bleibt die manuelle Kontrolle als Sicherheitsnetz bestehen; beidhändige
Überlappung und explizite Unsicherheitswarnungen sind mögliche spätere Erweiterungen.

#### P2.3 – Geschwindigkeiten pro Zug (implementiert)

Für die **bewegte Hand** werden pro Zug berechnet:

- Dauer;
- horizontaler, vertikaler und gesamter Weg;
- direkte Verschiebung und tatsächliche Pfadlänge;
- mittlere und maximale Geschwindigkeit;
- Zeitpunkt der maximalen Geschwindigkeit.

Für den **Körper** verwenden wir zunächst einen robusten Rumpfmittelpunkt aus Schultern und Hüften.
Gemessen werden Körperweg, vertikale Verschiebung sowie mittlere und maximale Geschwindigkeit.
Zusätzlich kann die Bewegung der ruhigen Hand relativ zum Körper beschrieben werden. Das ist ein
Hinweis auf eine Stützphase, aber keine Kraft- oder Zugleistungsmessung.

Die erste Version gibt Geschwindigkeiten in zwei Einheiten aus:

- `px/s` als direkte Bildmessung;
- `body_lengths/s` als grob körpergrößennormierte Vergleichsgröße.

Die Körperlänge wird pro Frame als anatomische Bildkette aus Schultermitte, Hüftmitte, Knie und
Knöchel geschätzt. Der Kletterer muss dafür nicht aufrecht stehen, weil die Kette segmentweise
summiert wird und ein gebeugtes Knie sie in der Bildebene nicht verkürzt. Normiert wird mit dem
Wert **pro Frame**, geglättet über etwa eine Sekunde: Tiefe ändert sich langsam, die
Verkürzung einzelner Frames ist dagegen verrauscht. Ein einziger Median über das ganze Video wäre
zu grob — am Referenzvideo schwankt die scheinbare Körperlänge zwischen 295 und 446 Pixeln, was
einzelne Züge um über zehn Prozent falsch normiert hätte. `BL` bleibt eine relative
Vergleichslänge, keine gemessene reale Körpergröße.

Echte `cm/s` oder `m/s` wären ohne Kalibrierung irreführend. Dafür brauchen wir später mindestens
eine bekannte Strecke in der Wandebene und möglichst eine statische Kamera. Perspektivische Tiefe
bleibt selbst dann eine Einschränkung.

Für Ableitungen wird eine eigene Glättung verwendet. Geschwindigkeit verstärkt kleine
Positionsfehler stark; einfach rohe Frame-Differenzen zu bilden wäre fachlich falsch. Die
Entrauschung leistet dabei der Positionsfilter (9-Frame-Median), nicht das Ableitungsfenster. Ein
Vergleich am Referenzvideo zeigt das deutlich: zwischen Radius 1 und Radius 10 bleibt der
Rauschboden in Ruhephasen nahezu unverändert, während die erfasste Spitzengeschwindigkeit von
100 auf 52 Prozent fällt. `speed_window_radius: 2` (rund 84 ms) hält deshalb die Spitzen fest,
ohne Rauschen einzuhandeln — das frühere Fenster von 251 ms hat kurze Züge stark verwischt.

Alle Wegspalten verwenden dieselbe Definition, nämlich die Summe der tatsächlichen Frameschritte.
`mean_speed_px_s` ist exakt `path_length_px / duration_seconds`.

Die Berechnung ist als reproduzierbare Cache-Stufe `80_move_metrics` implementiert. Eingaben sind
`pose_refined.parquet` und der aktuelle korrigierbare Stand in `annotations/.../moves.parquet`.
Ändern sich Zuggrenzen, entsteht beim nächsten Start automatisch ein neuer Metrik-Cache; Sapiens
wird dafür nicht erneut ausgeführt. Neben Parquet wird zur einfachen Einsicht dasselbe Ergebnis als
JSON gespeichert.

Der Player zeigt je nach Layout unter oder rechts neben dem Video eine Kurve mit Hand- und
Körpergeschwindigkeit pro Frame, beschrifteten Zeit- und BL/s-Achsen sowie den exakten Werten am
aktuellen Frame. Dazu stehen mittlere und maximale Geschwindigkeiten sowie die geschätzten Wege.
Eine zweite Kachelreihe zeigt `Hip rise`, `Hip below hand`, `Torso lead` und `Hand settles`, also
die Körperposition am Zugriff und den Rumpfvorlauf. Beim Vorlauf steht die Korrelation als
Kleintext darunter, damit ein schwach gestützter Wert erkennbar bleibt; ist er unbestimmt, zeigt
die Kachel `undefined` statt einer Zahl.
Die eigene framebasierte Video-Zeitleiste aktualisiert Bild, Frameanzeige und weißen Cursor
bereits während des Ziehens. Die Pfeiltasten und die beiden Frame-Buttons springen jeweils exakt
einen Videoframe; gehaltene Pfeiltasten warten auf den tatsächlich dargestellten Frame, bevor sie
zum nächsten springen. Die zugweisen
Zusammenfassungen stehen in `move_metrics.parquet`, die vollständigen Kurvenwerte in
`move_speed_timeline.parquet`. Am Referenzvideo ergeben sich:

| Zug | Ergebnis | Hand max. | Hand Ø | Körper max. | Körper Ø |
|---:|---|---:|---:|---:|---:|
| 1 | abgeschlossen | 5,50 KL/s | 0,97 KL/s | 2,14 KL/s | 0,51 KL/s |
| 2 | abgeschlossen | 5,82 KL/s | 0,54 KL/s | 0,33 KL/s | 0,11 KL/s |
| 3 | Sturz | 9,53 KL/s | 1,30 KL/s | 3,14 KL/s | 0,86 KL/s |

`KL/s` bedeutet geschätzte Körperlängen pro Sekunde. Der Sturz zeigt erwartungsgemäß die
höchste Körpergeschwindigkeit. Die absoluten `px/s` bleiben ebenfalls im Datensatz. Weil das eine
2D-Messung mit Modellunsicherheit ist, sind kleine Unterschiede nicht automatisch sportlich
bedeutsam; die Werte sind zunächst für den Vergleich von Zügen im selben Kamerabild gedacht.

##### Körperposition und Koordination

Zwei zusätzliche Gruppen zielen auf den Vergleich **zweier Versuche desselben Zugs**. Beide sind
bewusst so gewählt, dass sie gegen die Kameraposition unempfindlich sind: die eine misst
vertikale Verhältnisse in Körperlängen, die andere nur Zeiten.

`hand_settle_*` markiert den Frame, an dem die bewegte Hand zur Ruhe kommt — bei einem
abgeschlossenen Zug der Griff, bei einem Sturz der tiefste Punkt. Das ist absichtlich **nicht** das
Zugende, denn ein Zug schließt erst, wenn sich auch Körper und Beine beruhigt haben; die Hüfte dort
abzulesen würde die Haltung verfehlen, auf die es beim Zugriff ankam. Genommen wird der Beginn der
letzten ruhigen Strecke, weil eine Hand auf dem Weg oft kurz zögert, bevor sie sich festlegt.

Darauf setzen `hip_rise_body_lengths` (wie weit die Hüfte vom Zugbeginn bis zum Zugriff gestiegen
ist) und `hip_below_hand_body_lengths` (wie weit die Hüfte im Moment des Zugriffs unter der
Greifhand steht, also wie gestreckt der Körper war).

`coordination_lag_seconds` beantwortet, ob der Rumpf vor der Hand losgeht. Bewusst **ohne**
Schwellenwert: ein Kletterer ist vor einem Zug selten ruhig — am Referenzvideo bewegt sich der Rumpf
in der Sekunde vor zwei von drei Zügen mit 0,2 bis 1,2 KL/s, weil Füße gesetzt und Gewicht verlagert
wird. Ein Schwellenübertritt würde dort vor allem die Schwelle messen. Stattdessen werden die beiden
Geschwindigkeitskurven gegeneinander korreliert; ein positiver Wert bedeutet, der Rumpf war zuerst
dran. `coordination_correlation` steht daneben, damit ein schwach begründeter Versatz erkennbar
bleibt. Liegt das Korrelationsmaximum am Rand des durchsuchten Bereichs oder ist eine der beiden
Kurven flach, bleiben beide Felder `null` statt einen Randwert als Messung auszugeben.

Am Referenzvideo:

| Zug | Ergebnis | Hand ruht nach | Hüftanstieg | Hüfte unter Hand | Rumpfvorlauf |
|---:|---|---:|---:|---:|---:|
| 1 | abgeschlossen | 1,92 s | +0,32 KL | 1,10 KL | +17 ms (r=0,76) |
| 2 | abgeschlossen | 4,49 s | +0,29 KL | 1,04 KL | +936 ms (r=0,53) |
| 3 | Sturz | 2,69 s | **−0,64 KL** | 0,39 KL | −100 ms (r=0,78) |

Der Sturz trennt sich beim Hüftanstieg von selbst ab: als einziger Abschnitt verliert die Hüfte an
Höhe. Der Vorlauf von Zug 2 ist mit `r=0,53` allerdings schwach gestützt — dort bewegt sich die Hand
über Sekunden kaum, sodass die Korrelation breit und ihr Maximum unscharf ist.

#### P2.4 – Gelenkwinkel pro Zug

Als erste sinnvolle 2D-Winkel werden berechnet:

- linker und rechter Ellbogen;
- linke und rechte Schulter im Bild;
- linkes und rechtes Knie;
- linke und rechte Hüfte;
- Rumpfneigung.

Pro Zug speichern wir Winkel am Start und Ende sowie Minimum, Maximum und Bewegungsumfang. Frames
mit fehlenden oder zu unsicheren Gelenkpunkten werden nicht erfunden, sondern als ungültig
gekennzeichnet.

Alle Werte sind **2D-Bildwinkel**. Dreht sich der Kletterer zur Wand oder aus der Bildebene, sind sie
nicht identisch mit echten anatomischen 3D-Gelenkwinkeln.

#### P2.5 – Erweiterte Ergebnisansicht und Export

Die erste Geschwindigkeitskarte ist bereits in P2.3 enthalten. P2.5 erweitert den Player um:

- detaillierte Wege und den Höhengewinn;
- ausgewählte Winkel und Bewegungsumfang;
- Warnungen bei fehlenden oder unsicheren Daten.

Maschinenlesbare Ergebnisse bleiben in Parquet/JSON. Eine kompakte CSV kann zusätzlich exportiert
werden, damit einzelne Züge später einfach verglichen werden können.

### Empfohlene Reihenfolge

**P2.1** schlägt die Züge automatisch vor und spielt sie bequem vor/zurück ab. In **P2.2** werden
nur noch offensichtliche Fehler korrigiert und die Erkennung dagegen gemessen. Geschwindigkeit und
Winkel folgen erst, wenn die Segmente verlässlich sind; sonst würden präzise aussehende Zahlen dem
falschen Zug zugeordnet.

### Vorläufige Annahmen für Phase 2

- Die Kamera des ersten Referenzvideos gilt als statisch.
- Ein Zug wird primär durch Handbewegung definiert, unabhängig davon, ob er nach oben, seitlich oder
  nach unten geht.
- Dynos und zeitlich überlappende beidhändige Bewegungen können ein gemeinsamer Zug sein.
- Der Player und alle Auswertungen laufen lokal; es werden keine Videos hochgeladen.
- Automatische Zuggrenzen bleiben korrigierbar und erhalten eine Confidence.
- Physische Geschwindigkeit in Metern pro Sekunde wird erst nach einer expliziten Kalibrierung
  angeboten.
