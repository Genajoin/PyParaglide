# Руководство по развёртыванию Paraglidable

Этот документ описывает пошаговый процесс развёртывания проекта Paraglidable с использованием Docker.

## Предварительные требования

- Установленный Docker на хостовой системе
- Минимум 2ГБ свободного дискового пространства для загрузки данных
- Порт 8080 доступен на хосте (или измените проброс портов при необходимости)

## Шаги развёртывания

### 1. Сборка Docker-образа

```bash
cd /path/to/Paraglidable
docker build -t paraglidable2 docker/
```

Это создаёт Docker-образ со всеми необходимыми зависимостями:
- TensorFlow 1.15.0
- Python 3 с необходимыми пакетами
- Qt 5 для C++ tiler
- Apache HTTP server с PHP

**Время сборки:** ~5-10 минут (зависит от скорости сети)

### 2. Запуск Docker-контейнера

```bash
docker run -d --name paraglidable \
    -v /path/to/Paraglidable:/workspaces/Paraglidable \
    -p 8080:80 \
    paraglidable2 tail -f /dev/null
```

**Параметры:**
- `-v /path/to/Paraglidable:/workspaces/Paraglidable` - Монтирование директории проекта
- `-p 8080:80` - Проброс порта 8080 хоста на порт 80 контейнера
- `--name paraglidable` - Имя контейнера

### 3. Загрузка обучающих данных

```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/download_data.py
```

**Загружает:** ~200МБ исторических данных о полётах и погоде
**Расположение:** `/workspaces/Paraglidable/neural_network/bin/data/`

### 4. Загрузка elevation-тайлов

```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/download_elevation_tiles.py
```

**Загружает:** ~260МБ данных о высотах
**Расположение:** `/workspaces/Paraglidable/tiler/_cache/elevation/`

### 5. Сборка C++ Tiler

```bash
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/tiler/Tiler && qmake Tiler.pro && make"
```

**Результат:** исполняемый файл `/workspaces/Paraglidable/tiler/Tiler/Tiler`

### 6. Настройка веб-сервера Apache

```bash
# Обновление DocumentRoot
docker exec paraglidable bash -c "sudo sed -i 's|DocumentRoot /var/www/html|DocumentRoot /workspaces/Paraglidable/www|' /etc/apache2/sites-available/000-default.conf"

# Добавление директивы Directory
docker exec paraglidable bash -c "sudo bash -c 'cat >> /etc/apache2/sites-available/000-default.conf << EOF
<Directory /workspaces/Paraglidable/www>
    Options Indexes FollowSymLinks
    AllowOverride All
    Require all granted
</Directory>
EOF'"

# Включение модуля rewrite (требуется для .htaccess)
docker exec paraglidable bash -c "sudo a2enmod rewrite"

# Запуск Apache
docker exec paraglidable bash -c "sudo apache2ctl start"
```

### 7. Генерация прогноза

После развёртывания сайт пустой — нужно сгенерировать прогноз:

```bash
# Создать директорию для тайлов
docker exec paraglidable mkdir -p /workspaces/Paraglidable/www/data/tiles

# Запустить генерацию прогноза (запускать из neural_network!)
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/neural_network && python3 forecast.py"
```

**Что делает этот скрипт:**
1. Скачивает данные GFS с NOAA (~100-200МБ)
2. Обрабатывает GRIB файлы через pygrib
3. Запускает ML-прогноз через TensorFlow
4. Генерирует аргументы для tiler
5. Запускает C++ Tiler для создания PNG тайлов

**Время выполнения:** 10-20 минут

**Результат:**
- 27030+ PNG тайлов в `/workspaces/Paraglidable/www/data/tiles/`
- Прогноз на 10 дней вперёд
- Файлы `spots.json` с прогнозом для конкретных локаций

### 8. Доступ к приложению

После генерации прогноза откройте браузер: **http://localhost:8080/**

## Проверочные команды

```bash
# Проверка статуса контейнера
docker ps | grep paraglidable

# Проверка процессов Apache
docker exec paraglidable bash -c "ps aux | grep apache"

# Проверка веб-интерфейса
curl -s http://localhost:8080/ | head -20

# Проверка наличия исполняемого файла tiler
docker exec paraglidable ls -la /workspaces/Paraglidable/tiler/Tiler/Tiler

# Проверка наличия обучающих данных
docker exec paraglidable ls -la /workspaces/Paraglidable/neural_network/bin/data/

# Проверка сгенерированных тайлов
docker exec paraglidable ls -la /workspaces/Paraglidable/www/data/tiles/
docker exec paraglidable bash -c "find /workspaces/Paraglidable/www/data/tiles/ -name '*.png' | wc -l"

# Проверка spots.json
docker exec paraglidable bash -c "find /workspaces/Paraglidable/www/data/tiles/ -name 'spots.json'"
```

## Опционально: Загрузка фоновых тайлов

Для фоновых картографических тайлов (180МБ):

```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/download_background_tiles.py
```

## Работа внутри контейнера

```bash
# Вход в оболочку контейнера
docker exec -it paraglidable bash

# Обучение нейросети
cd /workspaces/Paraglidable/neural_network
python train.py

# Запуск прогноза
cd /workspaces/Paraglidable/neural_network
python forecast.py
```

## Управление контейнером

```bash
# Остановка контейнера
docker stop paraglidable

# Запуск контейнера (если уже создан)
docker start paraglidable

# Перезапуск Apache после перезапуска контейнера
docker exec paraglidable bash -c "sudo apache2ctl start"

# Удаление контейнера (осторожно!)
docker stop paraglidable && docker rm paraglidable
```

## Устранение неполадок

### Apache показывает ошибку 500 Internal Server Error

Включите модуль rewrite:
```bash
docker exec paraglidable bash -c "sudo a2enmod rewrite && sudo apache2ctl restart"
```

### Apache показывает стандартную страницу вместо Paraglidable

Проверьте конфигурацию DocumentRoot:
```bash
docker exec paraglidable cat /etc/apache2/sites-available/000-default.conf | grep DocumentRoot
```

Должно показывать: `DocumentRoot /workspaces/Paraglidable/www`

### Порт 8080 уже занят

Измените проброс портов:
```bash
docker run -d --name paraglidable \
    -v /path/to/Paraglidable:/workspaces/Paraglidable \
    -p 9090:80 \
    paraglidable2 tail -f /dev/null
```

Затем доступ по адресу: http://localhost:9090/

### Полная пересборка с нуля

```bash
docker stop paraglidable && docker rm paraglidable
docker rmi paraglidable2
# Затем начните с шага 1
```

## Быстрое развёртывание одной командой

Для опытных пользователей — сокращённая версия:

```bash
# Сборка и запуск
docker build -t paraglidable2 docker/ && \
docker run -d --name paraglidable -v $(pwd):/workspaces/Paraglidable -p 8080:80 paraglidable2 tail -f /dev/null

# Настройка
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/download_data.py && \
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/download_elevation_tiles.py && \
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/tiler/Tiler && qmake Tiler.pro && make" && \
docker exec paraglidable bash -c "sudo sed -i 's|DocumentRoot /var/www/html|DocumentRoot /workspaces/Paraglidable/www|' /etc/apache2/sites-available/000-default.conf && sudo a2enmod rewrite && sudo apache2ctl start"

# Генерация прогноза (после настройки)
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/neural_network && python3 forecast.py"
```

## Расположение файлов после развёртывания

| Компонент | Путь в контейнере |
|-----------|-------------------|
| Веб-файлы | `/workspaces/Paraglidable/www/` |
| Нейросеть | `/workspaces/Paraglidable/neural_network/` |
| Исполняемый файл tiler | `/workspaces/Paraglidable/tiler/Tiler/Tiler` |
| Обучающие данные | `/workspaces/Paraglidable/neural_network/bin/data/` |
| Elevation-тайлы | `/workspaces/Paraglidable/tiler/_cache/elevation/` |
| Сгенерированные тайлы | `/workspaces/Paraglidable/www/data/tiles/` |
| Конфигурация Apache | `/etc/apache2/sites-available/000-default.conf` |