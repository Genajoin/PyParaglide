// Background script (Service Worker) для обработки данных
console.log('[XContest Collector] Background script loaded');

// Импортируем db.js
importScripts('db.js');

// Инициализируем БД
const db = new FlightsDB();
let isInitialized = false;

// Инициализация БД
async function initDB() {
  if (!isInitialized) {
    await db.init();
    isInitialized = true;
    console.log('[XContest Collector] Database initialized');
  }
}

// Обработка сообщений от content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[XContest Collector] Received message:', message.type);

  if (message.type === 'API_RESPONSE') {
    handleApiResponse(message.url, message.data)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ success: false, error: error?.message || String(error) || 'Unknown error' }));
    return true; // Асинхронный ответ
  }

  if (message.type === 'GET_STATS') {
    getStats()
      .then(stats => sendResponse(stats))
      .catch(error => sendResponse({ error: error?.message || String(error) || 'Unknown error' }));
    return true;
  }

  if (message.type === 'EXPORT_DATA') {
    exportData()
      .then(data => sendResponse(data))
      .catch(error => sendResponse({ error: error?.message || String(error) || 'Unknown error' }));
    return true;
  }

  if (message.type === 'CLEAR_DATA') {
    clearData()
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: error?.message || String(error) || 'Unknown error' }));
    return true;
  }
});

// Обработка API ответа
async function handleApiResponse(url, data) {
  try {
    await initDB();

    console.log('[XContest Collector] Processing API response:', {
      url,
      itemsCount: data?.items?.length
    });

    // Сохраняем сырой API ответ
    try {
      await db.saveApiResponse(url, data);
    } catch (saveError) {
      console.warn('[XContest Collector] Failed to save raw API response:', saveError);
      // Продолжаем работу даже если не удалось сохранить сырой ответ
    }

    // Извлекаем полеты из ответа
    const flights = data?.items || [];

    if (flights.length === 0) {
      console.log('[XContest Collector] No flights in response');
      return { success: true, stats: { addedCount: 0, skippedCount: 0, total: 0, totalInDB: 0 } };
    }

    // Сохраняем полеты в БД
    console.log('[XContest Collector] Adding flights to DB...');
    const result = await db.addFlights(flights);
    console.log('[XContest Collector] Add flights result:', result);

    // Обновляем метаданные
    await db.setMetadata('lastUpdate', new Date().toISOString());
    await db.setMetadata('lastUrl', url);

    // Получаем общее количество полетов в БД
    const totalInDB = await db.getFlightsCount();

    console.log('[XContest Collector] Saved flights:', result);

    return {
      success: true,
      stats: {
        addedCount: result.addedCount,
        skippedCount: result.skippedCount,
        total: result.total,
        totalInDB: totalInDB
      }
    };
  } catch (error) {
    console.error('[XContest Collector] Error handling API response:', error);
    console.error('[XContest Collector] Error stack:', error.stack);
    return {
      success: false,
      error: error?.message || String(error) || 'Unknown error',
      errorName: error?.name,
      errorStack: error?.stack
    };
  }
}

// Получение статистики
async function getStats() {
  try {
    await initDB();
    const count = await db.getFlightsCount();
    const lastUpdate = await db.getMetadata('lastUpdate');
    const lastUrl = await db.getMetadata('lastUrl');
    
    return {
      count,
      lastUpdate,
      lastUrl
    };
  } catch (error) {
    console.error('[XContest Collector] Error getting stats:', error);
    throw error;
  }
}

// Экспорт данных
async function exportData() {
  try {
    await initDB();
    const flights = await db.getAllFlights(1000000); // Получить все
    
    return {
      success: true,
      data: flights,
      count: flights.length
    };
  } catch (error) {
    console.error('[XContest Collector] Error exporting data:', error);
    throw error;
  }
}

// Очистка данных
async function clearData() {
  try {
    await initDB();
    await db.clearFlights();
    await db.setMetadata('lastUpdate', null);
    await db.setMetadata('lastUrl', null);
    console.log('[XContest Collector] Data cleared');
  } catch (error) {
    console.error('[XContest Collector] Error clearing data:', error);
    throw error;
  }
}

console.log('[XContest Collector] Background script initialized');
