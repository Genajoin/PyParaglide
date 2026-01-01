# XContest Data Collector

Браузерное расширение для автоматического сбора данных полётов с XContest.org API.

## Возможности

- Автоматический перехват и сохранение данных из API запросов
- Хранение данных в IndexedDB (поддержка больших объемов >5MB)
- Контроль дублей по ID полёта
- Экспорт данных в JSON
- Статистика сбора
- Визуальные уведомления о сохранении

## Установка

### Chrome/Edge

1. Откройте `chrome://extensions/` (или `edge://extensions/`)
2. Включите "Режим разработчика" в правом верхнем углу
3. Нажмите "Загрузить распакованное расширение"
4. Выберите папку `extensions/xcontest_data_collector`

## Использование

1. Установите расширение
2. Откройте https://www.xcontest.org/world/en/flights/
3. Примените нужные фильтры (например, Slovenia + PG*)
4. Перемещайтесь по страницам пагинации
5. Расширение автоматически сохранит все данные из API запросов

### Визуальная обратная связь

При каждом сохранении данных появляется уведомление в правом верхнем углу:
```
✓ Данные сохранены
Добавлено: 100
Дубли: 0
Всего в БД: 500
```

### Управление данными

Кликните на иконку расширения для:
- **Обновить статистику** - показать количество собранных полётов
- **Экспорт в JSON** - скачать все данные в JSON файл
- **Очистить данные** - удалить все собранные данные

## Структура данных

Каждый полёт сохраняется со всеми полями из API:

```json
{
  "id": 6156451,
  "ident": "Lyahung/30.12.2025/09:18",
  "type": 1,
  "timeClaim": "2025-12-30T09:29:38Z",
  "pilot": {
    "id": 122192,
    "username": "Lyahung",
    "name": "A Hưng Lý",
    "countryIso": "VN"
  },
  "route": {
    "distance": 1.455,
    "points": 1.45
  },
  "takeoff": {
    "name": "Khau Pha"
  },
  "glider": {
    "name": "alpha 3",
    "classFAI": 3
  }
}
```

## Технические детали

- **Manifest V3** - современная версия Chrome Extensions
- **IndexedDB** - для хранения больших объёмов данных
- **Fetch Interception** - перехват API запросов в контексте страницы
- **Content Script + Background** - архитектура расширения

## Файлы

- `manifest.json` - конфигурация расширения
- `background.js` - service worker для обработки данных
- `content.js` - загрузка инжектируемого скрипта
- `injected.js` - перехват fetch/XHR запросов (инжектируется в страницу)
- `db.js` - работа с IndexedDB
- `popup.html/js` - UI для управления

## Отладка

Откройте консоль разработчика (F12) на странице XContest:
- Сообщения `[XContest Collector]` показывают работу расширения
- Вкладка "Application > IndexedDB > XContestFlightsDB" - данные в БД

## Проверка установки

### 1. Откройте XContest.org

Перейдите на:
```
https://www.xcontest.org/world/en/flights/
```

### 2. Откройте консоль разработчика

Нажмите **F12**

### 3. Примените фильтры

- Выберите страну (например, Slovenia)
- Выберите категорию планера (например, PG*)

### 4. Перейдите на страницу 2

В консоли должны появиться сообщения:
```
[XContest Collector] Content script loaded
[XContest Collector] Intercepted API request
[XContest Collector] ✓ Данные сохранены
```

## Экспорт данных

### Через UI

1. Кликните на иконку расширения
2. Нажмите **"Экспорт в JSON"**
3. Файл `xcontest_flights_YYYY-MM-DD.json` будет скачан

### Через Console

```javascript
const openDB = indexedDB.open('XContestFlightsDB', 1);
openDB.onsuccess = function() {
  const db = openDB.result;
  const tx = db.transaction('flights', 'readonly');
  const store = tx.objectStore('flights');
  const all = store.getAll();
  all.onsuccess = function() {
    console.log('Всего полётов:', all.result.length);
    copy(all.result); // Копирует в буфер обмена
  };
};
```

## Лицензия

GPL v3

## История изменений

См. [CHANGELOG.md](CHANGELOG.md)
