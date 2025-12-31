# Инструкция по установке и отладке XContest Data Collector

## Установка расширения

### 1. Откройте Chrome Extensions

Введите в адресной строке:
```
chrome://extensions/
```

### 2. Включите режим разработчика

В правом верхнем углу включите переключатель **"Режим разработчика"** (Developer mode)

### 3. Загрузите расширение

1. Нажмите кнопку **"Загрузить распакованное расширение"** (Load unpacked)
2. Выберите папку: `/home/gena/dev/Paraglidable/extensions/xcontest_data_collector`
3. Расширение должно появиться в списке

### 4. Проверьте, что расширение активно

- В списке расширений должна быть карточка "XContest Data Collector"
- Переключатель должен быть включен (синий)
- Иконка 🎈 должна быть видна в панели инструментов браузера

## Тестирование

### 1. Откройте XContest.org

Перейдите на:
```
https://www.xcontest.org/world/en/flights/
```

### 2. Откройте консоль разработчика

Нажмите **F12** или **Ctrl+Shift+I** (Cmd+Option+I на Mac)

### 3. Примените фильтры

- Выберите страну (например, Slovenia)
- Выберите категорию планера (например, PG*)

### 4. Проверьте консоль

В консоли должны появиться сообщения:

```
[XContest Collector] Content script loaded on: https://www.xcontest.org/...
[XContest Collector] Injected script running
[XContest Collector] Interception hooks installed
[XContest Collector] Injection script added to page
[XContest Collector] Content script initialized
```

### 5. Перейдите на страницу 2 или обновите страницу

При каждом API запросе должны появиться сообщения:

```
[XContest Collector] Fetch request: https://www.xcontest.org/api/data/?flights/...
[XContest Collector] ✓ Intercepted API request: https://www.xcontest.org/api/data/?flights/...
[XContest Collector] ✓ API Response received: {url: '...', itemsCount: 100, listInfo: {...}}
[XContest Collector] Received API data: {url: '...', itemsCount: 100}
[XContest Collector] Background response: {success: true, stats: {...}}
```

### 6. Проверьте уведомление

В правом верхнем углу страницы должно появиться зелёное уведомление:

```
✓ Данные сохранены
Добавлено: 100
Дубли: 0
Всего в БД: 100
```

### 7. Проверьте расширение

Кликните на иконку расширения в панели инструментов. Должно появиться окно с:

```
Собрано полетов: 100
Последнее обновление: 30.12.2025, 17:30:45
```

## Проверка данных в IndexedDB

### Вариант 1: Через DevTools

1. Откройте **DevTools** (F12)
2. Перейдите на вкладку **Application**
3. В левой панели: **Storage → IndexedDB → XContestFlightsDB**
4. Должны быть три объектных хранилища:
   - `flights` - собранные полеты
   - `metadata` - метаданные (lastUpdate, lastUrl)
   - `apiResponses` - сырые API ответы

5. Кликните на `flights` → должны быть записи с id полетов

### Вариант 2: Через Console

В консоли DevTools на странице XContest выполните:

```javascript
const openDB = indexedDB.open('XContestFlightsDB', 1);
openDB.onsuccess = function() {
  const db = openDB.result;
  const tx = db.transaction('flights', 'readonly');
  const store = tx.objectStore('flights');
  const count = store.count();
  count.onsuccess = function() {
    console.log('Всего полетов в БД:', count.result);
  };
};
```

## Отладка проблем

### Проблема 1: Нет сообщений в консоли

**Причина:** Content script не загружается

**Решение:**
1. Перезагрузите расширение в `chrome://extensions/`
2. Обновите страницу XContest (F5)
3. Проверьте, что расширение имеет права на `https://www.xcontest.org/*`

### Проблема 2: Сообщения есть, но "0 flights"

**Причина:** API запросы не перехватываются или формат данных не соответствует ожидаемому

**Решение:**
1. Проверьте в консоли, какие запросы логируются:
   ```
   [XContest Collector] Fetch request: <URL>
   ```
2. Убедитесь, что среди них есть запросы к `/api/data/`
3. Проверьте формат ответа - должно быть поле `data.items` с массивом полетов

### Проблема 3: Ошибки в консоли Background

**Как посмотреть:**
1. Перейдите в `chrome://extensions/`
2. Найдите "XContest Data Collector"
3. Кликните на "service worker" или "Посмотреть в фоновых страницах"
4. Откроется консоль background script

**Проверьте:**
```
[XContest Collector] Background script loaded
[XContest Collector] Database initialized
[XContest Collector] Received message: API_RESPONSE
[XContest Collector] Saved flights: {addedCount: 100, skippedCount: 0, total: 100}
```

### Проблема 4: Сайт использует XHR вместо fetch

**Симптом:** В логах только "Fetch request" но нет перехваченных API запросов

**Решение:** Проверьте есть ли сообщения "XHR open:" в консоли. Если есть - значит сайт использует XMLHttpRequest и это уже перехватывается.

## 🤖 Автоматический сбор данных

Если вам нужно собрать большой объем данных (например, за весь год), используйте автосбор:

### Как использовать автосбор

1. **Откройте popup расширения** (кликните на иконку 🎈)

2. **Прокрутите до раздела "Автосбор данных"**

3. **Настройте параметры:**
   - **Дата начала:** например, `2024-01-01`
   - **Дата окончания:** например, `2024-12-31`
   - **Страна:** (опционально) `SI` для Slovenia
   - **Категория планера:** (опционально) `FAI3` для PG*

4. **Нажмите "▶️ Начать автосбор"**

5. **Следите за прогрессом** в окне popup:
   ```
   Дата: 2024-06-15, обработано дат: 150, полетов: 12543, ошибок: 0
   ```

6. **Остановка (если нужно):**
   - Кнопка меняется на "⏸️ Остановить автосбор"
   - Кликните, чтобы прервать процесс

### Как это работает

Автосбор делает прямые API запросы к XContest:
```
https://www.xcontest.org/api/data/?flights/world/2024&lng=en&
  list[start]=0&list[num]=100&filter[date]=2024-06-15&
  filter[country]=SI&filter[detail_glider_catg]=FAI3
```

Для каждой даты в диапазоне:
1. Делается запрос к API
2. Данные сохраняются в IndexedDB
3. Дубли пропускаются автоматически
4. Прогресс обновляется каждые 2 секунды

### Скорость

- ~1 дата в секунду (с задержкой 1000ms между запросами)
- 30 дней = ~30 секунд
- 365 дней (год) = ~6 минут

### Ограничения

- API возвращает до 100 полетов за запрос
- Если за день >100 полетов, собираются только первые 100
- Для полного покрытия используйте фильтры (страна, категория)

### Пример: Сбор данных за 2024 год

```
Дата начала: 2024-01-01
Дата окончания: 2024-12-31
Страна: SI
Категория: FAI3
```

Результат: ~365 запросов, ~10,000-50,000 полетов (зависит от активности)

## Экспорт данных

### Через UI расширения

1. Кликните на иконку расширения
2. Нажмите **"📥 Экспорт в JSON"**
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
    console.log('Все полеты:', all.result);
    // Скопировать в буфер обмена:
    copy(all.result);
  };
};
```

## Очистка данных

### Через UI

1. Кликните на иконку расширения
2. Нажмите **"🗑️ Очистить данные"**
3. Подтвердите действие

### Через DevTools

1. **Application** → **IndexedDB** → **XContestFlightsDB**
2. Правой кнопкой → **Delete database**

## Обновление расширения после изменений

Если вы изменили код расширения:

1. Перейдите в `chrome://extensions/`
2. Найдите "XContest Data Collector"
3. Нажмите **кнопку обновления** (круговая стрелка)
4. Обновите страницу XContest (F5)

## Логи для отладки

Расширение создаёт подробные логи в трёх местах:

1. **Console страницы XContest** (F12 на странице):
   - Перехват fetch/XHR запросов
   - Отправка данных в background
   - Показ уведомлений

2. **Background Service Worker** (`chrome://extensions/` → service worker):
   - Получение сообщений
   - Сохранение в IndexedDB
   - Статистика

3. **Popup Console** (F12 в окне popup расширения):
   - Загрузка статистики
   - Экспорт данных
   - Очистка

Все логи начинаются с префикса `[XContest Collector]` для удобного поиска.
