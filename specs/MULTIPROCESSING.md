# Многопроцессинг в forecast.py

## Обзор

Документ описывает реализацию параллельной обработки дней прогноза для ускорения генерации тайлов.

## Текущая реализация (последовательная)

**Файл:** `neural_network/forecast.py`

```python
# Последовательная обработка дней
for day_idx, day_datetime in enumerate(forecast_days):
    # Скачать метеоданные
    # Сгенерировать прогноз
    # Создать тайлы
```

**Проблема:** Обработка 10 дней занимает ~10-20 минут на одном ядре CPU.

---

## Архитектура параллельной обработки

### Структура forecast.py

```python
class Forecast:
    def __init__(self):
        self.crops = [(93, 274, 0, 200), (93, 274, 1394, 1440)]
        self.last_forecast_time = ...
        # ... другие параметры

    def main(self):
        # Подготовка данных для параллельной обработки

        # 1. Список дней для обработки
        forecast_days = [...]
        nb_days = len(forecast_days)

        # 2. Параметры, одинаковые для всех дней
        shared_args = {
            'downloaded_forecasts_dir': self.downloaded_forecasts_dir,
            'tiles_dir': self.tiles_dir,
            'tiler_program': self.tiler_program,
            'tiler_arguments_filename': self.tiler_arguments_filename,
            'tiler_cache_dir': self.tiler_cache_dir,
            'geo_json_borders': self.geo_json_borders,
            'background_tiles_dir': self.background_tiles_dir,
            'skipped_tiles': self.skipped_tiles,
            'min_tiles_zoom': self.min_tiles_zoom,
            'max_tiles_zoom': self.max_tiles_zoom,
            'models_directory': self.models_directory,
            'problem_formulation': self.problem_formulation,
            'prediction_filename_for_tiler': self.prediction_filename_for_tiler,
            'crops': self.crops,
            'meteo_params': self.meteoParams,
            'grid_desc_predictions': self.grid_desc_predictions,
            'render_tiles': self.render_tiles,
            'debug_mode': self.DEBUG_MODE,
            'forced_meteo_files': self.forced_meteo_files,
        }

        # 3. Создание списка аргументов для каждого процесса
        args_list = []
        for day_idx, day_datetime in enumerate(forecast_days):
            strdate = day_datetime.strftime("%Y-%m-%d")
            args = (
                day_idx,
                day_datetime,
                strdate,
                self.last_forecast_time,
                self.forecast_time_dt,
                self.last_forecast_hour,
                self.destination_forecast_file,
                shared_args['downloaded_forecasts_dir'],
                # ... все остальные параметры
            )
            args_list.append(args)

        # 4. Запуск параллельной обработки
        nb_processes = min(10, nb_days, cpu_count())
        with Pool(processes=nb_processes) as pool:
            results = pool.map(process_single_day, args_list)

        # 5. Обработка результатов
        for day_idx, strdate, status, message in results:
            if status == "success":
                print(f"[OK] Day {strdate}")
            else:
                print(f"[FAILED] Day {strdate}: {message}")
```

---

## Worker функция для обработки одного дня

```python
def process_single_day(args):
    """
    Worker функция для параллельной обработки одного дня.

    Все параметры должны быть сериализуемы для multiprocessing.
    """
    # Распаковка аргументов
    (
        day_idx, day_datetime, strdate, last_forecast_time,
        forecast_time_dt, last_forecast_hour,
        destination_forecast_file, downloaded_forecasts_dir,
        tiles_dir, tiler_program, tiler_arguments_filename,
        tiler_cache_dir, geo_json_borders, background_tiles_dir,
        skipped_tiles, min_tiles_zoom, max_tiles_zoom,
        models_directory, problem_formulation,
        prediction_filename_for_tiler, crops, meteo_params, grid_desc_predictions,
        render_tiles, debug_mode, forced_meteo_files
    ) = args

    # Импорты внутри функции (важно для multiprocessing!)
    from inc.verbose import Verbose
    from inc.forecast_data import ForecastData
    from inc.predict import ForecastAndAnl, Predict
    from inc.dataset import GfsData, SpotsData, Spot
    from inc.model import ModelType
    from inc.trained_model import ModelContent
    from subprocess import call
    import numpy as np
    import os

    # Создать папку для тайлов этого дня
    tiles_dir_this_day = tiles_dir + "/" + strdate
    os.makedirs(tiles_dir_this_day, exist_ok=True)

    # ============================================================================
    # 1. Скачать/обновить метео файлы
    # ============================================================================

    meteo_files = []
    if forced_meteo_files is None:
        l_h = [(hh+24*day_idx-last_forecast_hour) for hh in [6, 12, 18]
               if (hh+24*day_idx-last_forecast_hour) >= 0]

        for h in l_h:
            forecast_datetime_with_hours = datetime.datetime(
                int(last_forecast_time[0:4]),
                int(last_forecast_time[4:6]),
                int(last_forecast_time[6:8]),
                int(last_forecast_time[8:10])
            )
            valid_datetime = forecast_datetime_with_hours + datetime.timedelta(hours=h)
            meteo_file = destination_forecast_file % valid_datetime.strftime("%Y-%m-%d-%H")
            meteo_files += [meteo_file]
    else:
        # Debug mode with forced meteo files
        meteo_files = ["/tmp/forced_meteo_06", "/tmp/forced_meteo_12", "/tmp/forced_meteo_18"]

    # Проверка существования метео файлов
    for mf in meteo_files:
        if not os.path.isfile(mf) or os.path.getsize(mf) <= 5000:
            Verbose.print_text(1, "[SKIP] Day %s: meteo files missing" % strdate)
            return (day_idx, strdate, "skipped", "meteo files missing")

    Verbose.print_text(1, "[PROCESSING] Day %s (pid=%s)" % (strdate, os.getpid()))

    # ============================================================================
    # 2. Прочитать погодные данные
    # ============================================================================

    distinct_latitudes, distinct_longitudes, meteo_matrix = \
        ForecastData.readWeatherData(meteo_files, crops)

    # ============================================================================
    # 3. Вычислить и сохранить прогноз для tiler
    # ============================================================================

    day_prediction_file = prediction_filename_for_tiler.replace(".txt", "_%s.txt" % strdate)

    ForecastAndAnl.compute_prediction_file_cells(
        ForecastAndAnl.compute_cells_forecasts(
            models_directory, problem_formulation, meteo_matrix
        ),
        day_prediction_file,
        distinct_latitudes,
        distinct_longitudes,
        np.copy(meteo_matrix),
        crops,
        meteo_params,
        grid_desc_predictions
    )

    # ============================================================================
    # 4. Вычислить прогноз для spots
    # ============================================================================

    day_spots_file = os.path.join(tiles_dir, strdate, "spots.json")

    predict = Predict(models_directory, ModelType.SPOTS, problem_formulation)
    predict.set_meteo_data(np.copy(meteo_matrix), GfsData().parameters_vector_all)

    # Собрать spots для каждого crop
    forecastCellsLine = {}
    line = 0
    for crop in crops:
        for iLat in range(crop[0], crop[1]):
            for iLon in range(crop[2], crop[3]):
                lat = distinct_latitudes[iLat]
                lon = distinct_longitudes[iLon]
                cell_coords = (lat, lon)
                forecastCellsLine[line] = cell_coords
                line += 1

    # Загрузить spots для региона
    from inc.bin_obj import BinObj
    filename_cells_and_spots = "Forecast_cellsAndSpots_" + "_".join(
        [str(crop[d]) for crop in crops for d in range(4)]
    )

    if not BinObj.exists(filename_cells_and_spots):
        spotsData = SpotsData()
        for line, cell_coords in forecastCellsLine.items():
            spotsData.load_spots(cell_coords[0], cell_coords[1])
        BinObj.save(spotsData, filename_cells_and_spots)
    else:
        spotsData = BinObj.load(filename_cells_and_spots)

    # Вычислить прогноз для spots
    spots = spotsData.get_all_spots()
    NN_X_spots = predict.get_X_for_spots(forecastCellsLine, spots)

    flyability_spots = predict.trainedModel.model.predict(NN_X_spots)

    # Сохранить spots.json
    spot_results = []
    for k, spot in enumerate(spots):
        lat, lon = spot.get_latlon()
        alt = spot.get_altitude()
        name = spot.get_name()
        fly = flyability_spots[0][k, 0, 0]
        spot_results.append({
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "name": name,
            "flyability": float(fly)
        })

    with open(day_spots_file, "w") as f:
        json.dump(spot_results, f)

    # ============================================================================
    # 5. Сгенерировать тайлы
    # ============================================================================

    if render_tiles:
        # Создать аргументный файл для tiler
        day_tiler_args_file = tiler_arguments_filename.replace(".txt", "_%s.txt" % strdate)

        ForecastAndAnl.generate_tiler_argument_file(
            day_tiler_args_file,
            day_prediction_file,
            tiles_dir_this_day,
            tiler_cache_dir,
            geo_json_borders,
            min_tiles_zoom,
            max_tiles_zoom,
            background_tiles_dir,
            skipped_tiles=skipped_tiles
        )

        # Запустить tiler
        call([tiler_program, day_tiler_args_file])

    return (day_idx, strdate, "success", "completed")
```

---

## Важные детали реализации

### 1. Импорты внутри worker функции

```python
# Правильно - импорты внутри функции
def process_single_day(args):
    from inc.verbose import Verbose
    from inc.predict import ForecastAndAnl
    # ...
```

**Причина:** При использовании `multiprocessing.Pool` функции выполняются в отдельных процессах. Модули должны быть импортированы внутри каждого процесса.

### 2. Сериализуемые аргументы

Все аргументы, передаваемые в `process_single_day()`, должны быть:
- Примитивными типами (int, float, str, bool)
- Списками/кортежами примитивов
- Numpy массивами
- **НЕ** могут быть объектами классов, открытыми файлами и т.д.

```python
# Правильно
args = (day_idx, strdate, crops_list, meteo_params)

# Неправильно
args = (self.verbose_instance, self.open_file_handle)
```

### 3. Количество процессов

```python
nb_processes = min(10, nb_days, cpu_count())
```

- **Ограничение по дням:** не создавать больше процессов чем дней
- **Ограничение CPU:** не перегружать систему
- **Жёсткий лимит:** максимум 10 процессов (можно настроить)

### 4. Возвращаемое значение

```python
return (day_idx, strdate, status, message)
```

Каждый worker возвращает кортеж с:
- `day_idx`: индекс дня (для сортировки результатов)
- `strdate`: строка даты (для логирования)
- `status`: "success", "skipped", "error"
- `message`: описание результата или ошибки

---

## Проверка запущенных процессов

**Исходный код (строки ~590-598):**

```python
# Проверка: не запущен ли уже forecast.py
import subprocess
output = subprocess.check_output(["ps", "aux"])
output = output.decode("utf-8")
running_processes = [(int(line.split()[1]), line.split()[10])
                     for line in output.split("\n")
                     if "forecast.py" in line and "defunct" not in line]
if len(running_processes) > 1:
    print(f"[WARNING] already running: {running_processes}")
    sys.exit(1)
```

**Проблема:** При использовании multiprocessing этот код даёт ложные срабатывания.

**Решение:** Закомментировать или удалить проверку при разработке multiprocessing.

---

## Тестирование

### 1. Тест с одним процессом

```python
nb_processes = 1  # Последовательная обработка
```

### 2. Тест с несколькими процессами

```python
nb_processes = min(3, nb_days)  # 3 параллельных процесса
```

### 3. Отладка worker функции

```python
# Прямой вызов worker функции (без multiprocessing)
args = args_list[0]  # Первый день
result = process_single_day(args)
print(result)
```

---

## Ожидаемое ускорение

| Конфигурация | Время на 10 дней |
|--------------|------------------|
| 1 процесс    | ~15-20 минут     |
| 4 процесса   | ~5-7 минут       |
| 10 процессов | ~2-4 минуты      |

**Ограничивающие факторы:**
- Скачивание GFS данных (IO, не CPU)
- Запись тайлов на диск (IO)
- Количество ядер CPU

---

## Проблемы и решения

### Проблема 1: PickleError при передаче аргументов

**Ошибка:** `TypeError: cannot pickle 'module' object`

**Решение:** Все импорты модулей должны быть внутри worker функции.

### Проблема 2: Память

**Ошибка:** Too many processes → OOM (Out of Memory)

**Решение:**
```python
# Уменьшить количество процессов
nb_processes = min(4, nb_days)
```

### Проблема 3: Конфликты при записи файлов

**Ошибка:** Два процесса пишут в один файл

**Решение:** Использовать уникальные имена файлов:
```python
day_prediction_file = prediction_filename_for_tiler.replace(".txt", "_%s.txt" % strdate)
```

---

## Следующие шаги для реализации

1. **Откатить текущие изменения** в `forecast.py`
2. **Создать ветку** для multiprocessing: `git checkout -b feature/multiprocessing`
3. **Постепенная реализация:**
   - Шаг 1: Вынести код обработки дня в отдельную функцию
   - Шаг 2: Протестировать последовательный вызов функции
   - Шаг 3: Добавить multiprocessing.Pool с 1 процессом
   - Шаг 4: Увеличить количество процессов
4. **Тестирование** на Docker контейнере
5. **Benchmarking**: измерить ускорение
6. **Merge** в master после успешного тестирования

---

## Полезные команды

```bash
# Запуск forecast.py с multiprocessing
cd neural_network/
python forecast.py

# Мониторинг процессов
htop
# или
top

# Проверка запущенных Python процессов
ps aux | grep python

# Проверка использования CPU
top -p $(pgrep -d',' forecast.py)
```

---

## Статус

**Текущий статус:** В разработке

**Следующее действие:** Откатить изменения и сфокусироваться на переобучении модели для расширения области до 37°E.
