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

// Хранилище состояния автоматизации
let automationState = {
  isRunning: false,
  config: null,
  progress: null
};

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

  // === Обработка команд автоматизации ===

  if (message.type === 'START_AUTOMATION') {
    // Используем message.tabId, потому что sender.tab?.id undefined когда отправитель - popup
    startAutomation(message.config, message.tabId)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ success: false, error: error?.message || String(error) }));
    return true;
  }

  if (message.type === 'STOP_AUTOMATION') {
    stopAutomation(message.tabId)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ success: false, error: error?.message || String(error) }));
    return true;
  }

  if (message.type === 'GET_AUTOMATION_STATUS') {
    sendResponse({ success: true, state: automationState });
    return true;
  }

  if (message.type === 'AUTOMATION_PROGRESS') {
    automationState.progress = message.data;
    saveAutomationProgress(message.data)
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: error?.message || String(error) }));
    return true;
  }

  if (message.type === 'AUTOMATION_COMPLETE') {
    automationState.isRunning = false;
    clearAutomationProgress()
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: error?.message || String(error) }));
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
    console.log('[XContest Collector] Starting export...');

    const flights = await db.getAllFlights(1000000); // Получить все
    console.log(`[XContest Collector] Retrieved ${flights.length} flights from DB`);

    // Создаём JSON прямо в background (не передаём через message)
    console.log('[XContest Collector] Stringifying JSON...');
    const jsonString = JSON.stringify(flights, null, 2);
    console.log(`[XContest Collector] JSON size: ${(jsonString.length / 1024 / 1024).toFixed(2)} MB`);

    // Создаём Data URL (Service Worker не поддерживает Blob URL)
    console.log('[XContest Collector] Creating Data URL...');
    const dataUrl = 'data:application/json;charset=utf-8,' + encodeURIComponent(jsonString);
    const dateStr = new Date().toISOString().split('T')[0];

    // Запускаем скачивание
    console.log('[XContest Collector] Starting download...');
    await chrome.downloads.download({
      url: dataUrl,
      filename: `xcontest_flights_${dateStr}.json`,
      saveAs: true
    });

    console.log('[XContest Collector] Export complete');

    return {
      success: true,
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

// === Функции для работы с автоматизацией ===

/**
 * Запуск автоматизации
 * @param {Object} config - конфигурация автоматизации
 * @param {number} tabId - ID вкладки
 */
async function startAutomation(config, tabId) {
  if (!tabId) {
    throw new Error('Tab ID is required');
  }

  console.log('[XContest Collector] Starting automation with config:', config);

  automationState.isRunning = true;
  automationState.config = config;

  // Сохранить конфигурацию в chrome.storage
  await chrome.storage.local.set({
    automationConfig: config,
    automationProgress: {
      currentDate: config.currentDate || config.startDate,
      currentPage: config.currentPage || 1
    }
  });

  // Отправить команду в content script
  const response = await chrome.tabs.sendMessage(tabId, {
    type: 'START_AUTOMATION',
    config: config
  });

  return response;
}

/**
 * Остановка автоматизации
 * @param {number} tabId - ID вкладки
 */
async function stopAutomation(tabId) {
  if (!tabId) {
    throw new Error('Tab ID is required');
  }

  console.log('[XContest Collector] Stopping automation');

  automationState.isRunning = false;

  // Очистить прогресс
  await chrome.storage.local.remove(['automationConfig', 'automationProgress']);

  // Отправить команду в content script
  const response = await chrome.tabs.sendMessage(tabId, {
    type: 'STOP_AUTOMATION'
  });

  return response;
}

/**
 * Сохранение прогресса автоматизации
 * @param {Object} progress - данные прогресса
 */
async function saveAutomationProgress(progress) {
  console.log('[XContest Collector] Saving automation progress:', progress);

  await chrome.storage.local.set({
    automationProgress: {
      currentDate: progress.config?.currentDate,
      currentPage: progress.config?.currentPage,
      processedDates: progress.stats?.processedDates
    }
  });
}

/**
 * Очистка прогресса автоматизации
 */
async function clearAutomationProgress() {
  console.log('[XContest Collector] Clearing automation progress');

  await chrome.storage.local.remove(['automationConfig', 'automationProgress']);
}

/**
 * Получение доступных дат из content script
 * @param {number} tabId - ID вкладки
 */
async function getAvailableDates(tabId) {
  if (!tabId) {
    throw new Error('Tab ID is required');
  }

  console.log('[XContest Collector] Getting available dates from tab', tabId);

  const response = await chrome.tabs.sendMessage(tabId, {
    type: 'GET_AVAILABLE_DATES'
  });

  return response;
}

console.log('[XContest Collector] Background script initialized');
