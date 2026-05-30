[English](../README.md) · **Українська**

# IoT Edge AI Anomaly Detector

> Легковагова система виявлення аномалій в IoT-метриках на основі GRU — **1 516 параметрів**, **0.31 мс інференції**, **F1 = 0.945** на незалежному holdout-тесті. Розрахована на edge-розгортання без хмарної залежності.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

Архітектура знайдена трифазним еволюційним пошуком зі строгим no-leak протоколом — dev-test та holdout-test набори лишаються незалежними протягом усього процесу, тому отримана оцінка відображає реальну генералізацію, а не selection bias.

## Чому

- **Маленька.** 1 516 навчальних параметрів; навчений пакет моделі важить <30 КБ.
- **Швидка.** 0.31 мс на одне вікно на CPU; GPU не потрібен.
- **Точна.** F1 = 0.945 / Precision = 0.918 / Recall = 0.973 / ROC-AUC = 0.995.
- **Чесна.** Holdout повністю ізольований від циклу пошуку — виміряний bias 0.016 F1.
- **Самодостатня.** Генератор синтетичних даних, еволюційний пошук, калібровка, дашборди — все в одному пакеті.
- **Production-style CLI.** 9 субкоманд, `rich`-форматування виводу, повне збереження/завантаження пакета моделі.

## Завантаження

Готові самодостатні збірки публікуються на сторінці [Releases](https://github.com/1minEpowMinX/iot-edge-ai-anomaly-detector-public/releases) — **встановлення Python не потрібне**. Для кожної платформи окремий архів за схемою `iot-edge-ai-anomaly-detector-{платформа}.7z`:

| Платформа | Файл |
|---|---|
| Windows x64 | `iot-edge-ai-anomaly-detector-win64.7z` |
| Linux x64 | `iot-edge-ai-anomaly-detector-linux64.7z` |
| macOS | `iot-edge-ai-anomaly-detector-macos.7z` |

Кожен архів містить виконуваний файл та теку `_internal/` з усіма залежностями (PyInstaller onedir). **Тримайте виконуваний файл і теку `_internal/` разом в одній директорії.**

Перевірка завантаження за `SHA256SUMS.txt`, доданим до релізу:

```bash
# Linux / macOS
sha256sum iot-edge-ai-anomaly-detector-linux64.7z
# Windows PowerShell
(Get-FileHash .\iot-edge-ai-anomaly-detector-win64.7z -Algorithm SHA256).Hash
```

## Швидкий старт (готова збірка)

Розпакуйте архів, відкрийте термінал у теці з розпакованою програмою та виконайте:

```bash
# Windows
iot-edge-ai-anomaly-detector.exe demo
iot-edge-ai-anomaly-detector.exe demo --quick     # ~5 с прискорений варіант
iot-edge-ai-anomaly-detector.exe --help

# Linux / macOS — ті самі команди, без .exe та з префіксом ./
./iot-edge-ai-anomaly-detector demo
```

> У прикладах нижче використано ім'я виконуваного файлу для Windows. У Linux/macOS команди ідентичні — приберіть `.exe` та додайте префікс `./`.

Усі артефакти (дашборди, пакет моделі, метаінформація) пишуться в `artifacts/`.

## Запуск з вихідного коду (альтернатива)

Якщо ви бажаєте запускати з вихідного коду або працюєте на платформі, для якої немає збірки:

```bash
git clone https://github.com/1minEpowMinX/iot-edge-ai-anomaly-detector-public
cd iot-edge-ai-anomaly-detector-public
pip install -r requirements.txt
python main.py demo
```

Вимоги: Python 3.10+, PyTorch 2.0+, scikit-learn, NumPy, pandas, matplotlib, psutil, rich.

> При запуску з вихідного коду замінюйте `iot-edge-ai-anomaly-detector.exe` у будь-якій команді нижче на `python main.py`.

## CLI

| Команда   | Призначення |
|-----------|-------------|
| `demo`    | Тренує переможця еволюційного пошуку на holdout-даних. Головна демонстрація. |
| `train`   | Тренування з налаштовуваними гіперпараметрами через CLI. |
| `infer`   | Інференс на CSV-файлі зі збереженою моделлю. |
| `live`    | Real-time моніторинг хоста через psutil — демонстрація inference-pipeline. |
| `collect` | Збирає реальні метрики хоста у CSV. |
| `search`  | Трифазний еволюційний пошук (GA → shortlist → holdout retrain). |
| `compare` | Порівняння GRU vs LSTM vs MovingAverage на однакових даних. |
| `ablate`  | Аблація: 5-метрична підмножина (мінімальна за ТЗ) vs повний 12-метричний набір. |
| `sweep`   | Розгортка `window_size` та/або `hidden_size`. |

Глобальні прапорці: `--version`, `-v/--verbose`, `-q/--quiet`.

## Приклади

```bash
iot-edge-ai-anomaly-detector.exe demo
iot-edge-ai-anomaly-detector.exe train --epochs 100 --lr 1e-3 --hidden 16 --window 40
iot-edge-ai-anomaly-detector.exe compare              # GRU vs LSTM vs MA
iot-edge-ai-anomaly-detector.exe ablate               # 5 vs 12 ознак
iot-edge-ai-anomaly-detector.exe sweep --axis window_size
iot-edge-ai-anomaly-detector.exe search --quick       # швидкий еволюційний пошук
iot-edge-ai-anomaly-detector.exe collect --duration 60 -o my.csv
iot-edge-ai-anomaly-detector.exe infer --model artifacts/ --data my.csv
```

## Як це працює

```mermaid
flowchart TB
    M["метрики хоста<br>(12 каналів)"] --> S["MinMax скейлер"]
    S --> W["Sliding window<br>(W = 40)"]
    W --> P{"EMA<br>передфільтр"}
    P -- явно НОРМАЛЬНІ<br>(~25-30 % вікон) --> FAST["Швидкий шлях"]
    P -- невизначені /<br>підозрілі --> G["GRU forward<br>(1 516 параметрів)"]
    G --> R["прогноз − реальність"]
    R --> SC["оцінка аномальності<br>s_t = MAE по вікну"]
    SC --> T{"s_t &gt; τ ?"}
    T -- ні --> N(["НОРМА"])
    T -- так --> A(["АНОМАЛІЯ"])
    FAST --> N
     P:::decision
     FAST:::fast
     G:::gru
     R:::gru
     SC:::gru
     T:::decision
     A:::anomalyEnd
     N:::normalEnd
    classDef fast fill:#dff5e1,stroke:#3c9,color:#000
    classDef gru fill:#fff0d4,stroke:#c83,color:#000
    classDef decision fill:#e6e9ff,stroke:#55a,color:#000
    classDef anomalyEnd fill:#ffd9d9,stroke:#a44,color:#000,font-weight:bold
    classDef normalEnd fill:#dff5e1,stroke:#3c9,color:#000
```

GRU тренується прогнозувати наступний крок. Аномалії проявляються як значні відхилення прогнозу від реального значення (MAE-based score). Легковаговий EMA-передфільтр відсіює явно нормальні вікна, щоб зекономити обчислення — до GRU доходять тільки `UNCERTAIN`/`ANOMALY` кандидати. Поріг прийняття рішення τ автоматично калібрується на окремому калібрувальному наборі для максимізації F1 (метод Auto-F1).

## Розгортання на реальному пристрої

Модель у комплекті навчена на **синтетичних даних**, тому на реальному обладнанні можливі хибні спрацювання через зсув розподілу (distribution shift). Для робочого використання:

1. Зібрати метрики цільового пристрою: `iot-edge-ai-anomaly-detector.exe collect --duration 3600 -o real.csv`
2. Перенавчити на цих даних: `iot-edge-ai-anomaly-detector.exe train --epochs 200`
3. Використати отриманий пакет з `artifacts/` (`model.pt` + `scaler_*.npy` + `meta.json`).

Пакет портативний — завантажується через `src.artifacts.load_bundle()` на цільовому пристрої.

## Результати

### Holdout-метрики (seed = 999, ніколи не бачені під час пошуку)

| Метрика             | Значення        |
|---------------------|----------------:|
| **F1**              | **0.9449**      |
| Precision           | 0.9184          |
| Recall              | 0.9730          |
| ROC-AUC             | 0.995           |
| Параметри           | 1 516           |
| Час інференції      | 0.31 мс / вікно |
| Розмір пакета моделі | ~28 КБ         |

**Геном переможця:** `window_size=40, hidden_size=8, num_layers=3, dropout=0.4, lr=3e-3`.

### Порівняння моделей (`compare`)

| Модель         | Precision | Recall  | F1        | Параметри | Інференс  |
|----------------|----------:|--------:|----------:|----------:|----------:|
| **GRU**        | 0.918     | 0.973   | **0.945** | 1 516     | 0.31 мс   |
| LSTM           | 0.741     | 1.000   | 0.851     | 6 348     | 0.29 мс   |
| Moving Average | 0.442     | 0.984   | 0.610     | 0         | 0.004 мс  |

### Перевірка на selection bias

- F1 на dev-test (під час пошуку): **0.961**
- F1 на holdout (ізольований): **0.945**
- **Розрив: 0.016 F1** — у межах очікуваного для правильно захищеного no-leak протоколу.

## Архітектурні рішення

| Рішення | Обґрунтування |
|---|---|
| **GRU** замість LSTM | Менше параметрів при тому ж F1; емпірично підтверджено через `compare`. |
| **Linear readout** | Без активації MLP-head колапсує в один Linear; з активацією покращення F1 не зафіксовано. |
| **Huber loss** (навчання) | Стійкий до випадкових викидів у нібито нормальних тренувальних даних. |
| **MAE** (anomaly score) | Лінійний відгук підсилює контраст між нормальним шумом та аномальними сплесками — на відміну від Huber, який його гасить. |
| **AdamW + ReduceLROnPlateau** | Decoupled weight decay + адаптивний learning rate. |
| **Early stopping із patience-reset при lr-drop** | Не вбиває моделі, які ще можуть навчатися на нижчому lr. |
| **Асиметричний EMA-передфільтр** | По швидкому шляху йдуть тільки явно нормальні вікна; підозрілі завжди доходять до GRU. Зберігає recall, економить ~25–30 % forward-проходів. |
| **Auto-F1 калібрування порога** | Поріг підбирається на окремому labelled-наборі, а не ставиться довільно на 95-й перцентиль. |
| **No-leak протокол** | Holdout test (seed=999) ніколи не використовується під час пошуку. Dev-test (seed=123) — тільки для fitness. |
| **12 метрик** замість 5 | Канали диску / swap / процесів критичні для I/O штормів, витоків пам'яті, fork-bomb; аблація підтверджує приріст +0.20 F1. |
| **Еволюційний пошук** | Знаходить менші й кращі архітектури, ніж ручний тюнінг (1.5K параметрів vs 12K, +0.04 F1). |

## Структура проєкту

```
main.py                    CLI entrypoint (python main.py)
src/                       Production-пакет
├── __init__.py            Публічний API + __version__
├── cli.py                 argparse + 9 субкоманд
├── _ui.py                 rich UI з plain-text fallback
├── config.py              AppConfig + усі sub-config'и
├── data.py                Генератор синтетики + DataModule + scaler + вікна
├── model.py               GRUNet / LSTMNet
├── predictors.py          BasePredictor + Torch/MovingAverage predictors
├── losses.py              Фабрика Huber / MSE / MAE
├── prefilter.py           EMA / MA каскадний передфільтр
├── detector.py            AnomalyDetector + prf1 + roc_pr_curves (scikit-learn)
├── pipeline.py            Pipeline + RunResult
├── reporter.py            Reporter (консоль + дашборди)
├── visualize.py           matplotlib плотери
├── artifacts.py           Збереження / завантаження пакета моделі
└── experiments/           Лабораторні + production раннери
    ├── evolution.py       Genome / SearchSpace / GA / shortlist / retrain
    ├── search.py          Трифазна оркестрація
    ├── demo.py            Showcase-раннер
    ├── train.py           Configurable training runner
    ├── infer.py           CSV-інференс
    ├── live.py            Real-time psutil monitor
    ├── collect.py         psutil-семплер
    ├── comparison.py      Порівняння моделей
    ├── ablation.py        Аблація набору ознак
    └── sweep.py           Гіперпараметричні розгортки
```

## Контекст

Проєкт спершу створювався як інженерна дипломна робота з тематики edge AI / виявлення аномалій у часових рядах. Методологічний фокус на:

- Трактуванні задачі виявлення аномалій як **прогнозу наступного кроку з порогом на залишку** (а не реконструкції чи класифікації).
- Наскрізному **no-leak дослідницькому протоколі**, що відокремлює підбір гіперпараметрів від фінальної оцінки.
- Демонстрації того, що **еволюційний пошук знаходить компактні, придатні для edge архітектури**, які перевершують ручний тюнінг.

Якщо щось із цього корисне у твоїй роботі — форкай, цитуй або відкривай issue.

## Ліцензія

[MIT](../LICENSE) © 2026 1minEpowMinX.
