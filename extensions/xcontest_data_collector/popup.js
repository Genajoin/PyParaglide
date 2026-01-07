// Popup script для управления расширением
document.addEventListener('DOMContentLoaded', async () => {
  console.log('[XContest Collector] Popup loaded');

  // Элементы UI
  const flightsCountEl = document.getElementById('flightsCount');
  const lastUpdateEl = document.getElementById('lastUpdate');
  const statusEl = document.getElementById('status');
  const refreshBtn = document.getElementById('refreshBtn');
  const exportBtn = document.getElementById('exportBtn');
  const clearBtn = document.getElementById('clearBtn');

  // Загрузка статистики
  async function loadStats() {
    try {
      const stats = await chrome.runtime.sendMessage({ type: 'GET_STATS' });

      flightsCountEl.textContent = stats.count || 0;

      if (stats.lastUpdate) {
        const date = new Date(stats.lastUpdate);
        lastUpdateEl.textContent = date.toLocaleString('ru-RU');
      } else {
        lastUpdateEl.textContent = 'Нет данных';
      }

      showStatus('Статистика обновлена', 'success');
    } catch (error) {
      console.error('[XContest Collector] Error loading stats:', error);
      showStatus('Ошибка загрузки статистики', 'warning');
    }
  }

  // Показать статус
  function showStatus(message, type = 'info') {
    statusEl.textContent = message;
    statusEl.className = `status status-${type}`;
    statusEl.style.display = 'block';

    setTimeout(() => {
      statusEl.style.display = 'none';
    }, 3000);
  }

  // Обработчики кнопок
  refreshBtn.addEventListener('click', async () => {
    refreshBtn.disabled = true;
    refreshBtn.textContent = '⏳ Обновление...';

    await loadStats();

    refreshBtn.disabled = false;
    refreshBtn.textContent = '🔄 Обновить статистику';
  });

  exportBtn.addEventListener('click', async () => {
    try {
      exportBtn.disabled = true;
      exportBtn.textContent = '⏳ Экспорт...';

      // Background script теперь обрабатывает весь экспорт сам
      const result = await chrome.runtime.sendMessage({ type: 'EXPORT_DATA' });

      if (result.success) {
        showStatus(`Экспортировано ${result.count} полетов`, 'success');
      } else {
        showStatus('Ошибка экспорта', 'warning');
      }
    } catch (error) {
      console.error('[XContest Collector] Error exporting:', error);
      showStatus('Ошибка экспорта: ' + (error.message || 'Unknown error'), 'warning');
    } finally {
      exportBtn.disabled = false;
      exportBtn.textContent = '📥 Экспорт в JSON';
    }
  });

  clearBtn.addEventListener('click', async () => {
    if (!confirm('Вы уверены, что хотите удалить все собранные данные?')) {
      return;
    }

    try {
      clearBtn.disabled = true;
      clearBtn.textContent = '⏳ Очистка...';

      const result = await chrome.runtime.sendMessage({ type: 'CLEAR_DATA' });

      if (result.success) {
        showStatus('Данные очищены', 'success');
        await loadStats();
      } else {
        showStatus('Ошибка очистки', 'warning');
      }
    } catch (error) {
      console.error('[XContest Collector] Error clearing:', error);
      showStatus('Ошибка очистки', 'warning');
    } finally {
      clearBtn.disabled = false;
      clearBtn.textContent = '🗑️ Очистить данные';
    }
  });

  // === Автоматизация ===

  // Элементы UI автоматизации
  const startDateSelect = document.getElementById('startDateSelect');
  const endDateSelect = document.getElementById('endDateSelect');
  const automationProgress = document.getElementById('automationProgress');
  const progressFill = document.getElementById('progressFill');
  const progressStatus = document.getElementById('progressStatus');
  const startAutomationBtn = document.getElementById('startAutomationBtn');
  const stopAutomationBtn = document.getElementById('stopAutomationBtn');
  const resumeAutomationBtn = document.getElementById('resumeAutomationBtn');

  // Загрузка доступных дат
  async function loadAvailableDates(retryCount = 0) {
    const maxRetries = 3;
    const retryDelay = 1000;

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab || !tab.url || !tab.url.includes('xcontest.org')) {
        startDateSelect.innerHTML = '<option>Откройте XContest</option>';
        endDateSelect.innerHTML = '<option>Откройте XContest</option>';
        startAutomationBtn.disabled = true;
        return;
      }

      console.log('[Popup] Requesting available dates from tab', tab.id);

      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'GET_AVAILABLE_DATES'
      });

      console.log('[Popup] Response from tab:', response);

      if (response.success && response.dates && response.dates.length > 0) {
        populateDateSelects(response.dates);
        startAutomationBtn.disabled = false;
      } else if (retryCount < maxRetries) {
        // Retry - возможно automation.js еще не загружен
        console.log(`[Popup] Retrying... (${retryCount + 1}/${maxRetries})`);
        startDateSelect.innerHTML = `<option>Загрузка... (попытка ${retryCount + 1})</option>`;
        endDateSelect.innerHTML = `<option>Загрузка... (попытка ${retryCount + 1})</option>`;

        await new Promise(resolve => setTimeout(resolve, retryDelay));
        return loadAvailableDates(retryCount + 1);
      } else {
        const errorMsg = response.error || 'Нет доступных дат';
        console.error('[Popup] Failed to load dates:', errorMsg);
        startDateSelect.innerHTML = `<option>${errorMsg}</option>`;
        endDateSelect.innerHTML = `<option>${errorMsg}</option>`;
        startAutomationBtn.disabled = true;
        showStatus(`Ошибка загрузки дат: ${errorMsg}`, 'warning');
      }
    } catch (error) {
      console.error('[Popup] Error loading dates:', error);

      if (retryCount < maxRetries) {
        // Retry
        console.log(`[Popup] Retrying after error... (${retryCount + 1}/${maxRetries})`);
        startDateSelect.innerHTML = `<option>Загрузка... (попытка ${retryCount + 1})</option>`;
        endDateSelect.innerHTML = `<option>Загрузка... (попытка ${retryCount + 1})</option>`;

        await new Promise(resolve => setTimeout(resolve, retryDelay));
        return loadAvailableDates(retryCount + 1);
      } else {
        startDateSelect.innerHTML = '<option>Перезагрузите страницу</option>';
        endDateSelect.innerHTML = '<option>Перезагрузите страницу</option>';
        startAutomationBtn.disabled = true;
        showStatus('Ошибка: перезагрузите страницу XContest', 'warning');
      }
    }
  }

  function populateDateSelects(dates) {
    // Очистить существующие опции
    startDateSelect.innerHTML = '';
    endDateSelect.innerHTML = '';

    // Добавить даты
    dates.forEach(date => {
      const optionStart = document.createElement('option');
      optionStart.value = date.value;
      optionStart.textContent = `${date.text}`;
      startDateSelect.appendChild(optionStart);

      const optionEnd = document.createElement('option');
      optionEnd.value = date.value;
      optionEnd.textContent = `${date.text}`;
      endDateSelect.appendChild(optionEnd);
    });

    // Выбрать первую и последнюю дату по умолчанию
    if (dates.length > 0) {
      startDateSelect.selectedIndex = 0;
      endDateSelect.selectedIndex = Math.min(6, dates.length - 1); // По умолчанию 7 дней
    }
  }

  // Проверка на наличие сохраненного прогресса
  async function checkSavedProgress() {
    const storage = await chrome.storage.local.get(['automationConfig', 'automationProgress']);

    if (storage.automationConfig && storage.automationProgress) {
      // Показать кнопку "Продолжить"
      resumeAutomationBtn.style.display = 'block';
      startAutomationBtn.textContent = '▶️ Запустить заново';

      showStatus(`Найден сохраненный прогресс: ${storage.automationProgress.currentDate}`, 'info');
    }
  }

  // Запуск автоматизации
  startAutomationBtn.addEventListener('click', async () => {
    const startDate = startDateSelect.value;
    const endDate = endDateSelect.value;

    if (!startDate || !endDate) {
      showStatus('Выберите диапазон дат', 'warning');
      return;
    }

    const config = {
      startDate,
      endDate,
      currentDate: startDate,
      currentPage: 1,
      totalPages: 0
    };

    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab || !tab.id) {
        showStatus('Не удалось найти активную вкладку', 'warning');
        return;
      }

      // Очистить старый прогресс
      await chrome.storage.local.remove(['automationConfig', 'automationProgress']);

      await chrome.runtime.sendMessage({
        type: 'START_AUTOMATION',
        config: config,
        tabId: tab.id
      });

      // Обновить UI
      automationProgress.style.display = 'block';
      startAutomationBtn.style.display = 'none';
      stopAutomationBtn.style.display = 'block';
      resumeAutomationBtn.style.display = 'none';
      progressFill.style.width = '0%';
      progressStatus.textContent = 'Запуск...';

      showStatus('Автоматизация запущена', 'success');
      startProgressPolling();
    } catch (error) {
      showStatus('Ошибка запуска: ' + error.message, 'warning');
      console.error(error);
    }
  });

  // Остановка автоматизации
  stopAutomationBtn.addEventListener('click', async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab || !tab.id) {
        showStatus('Не удалось найти активную вкладку', 'warning');
        return;
      }

      await chrome.runtime.sendMessage({
        type: 'STOP_AUTOMATION',
        tabId: tab.id
      });

      automationProgress.style.display = 'none';
      startAutomationBtn.style.display = 'block';
      stopAutomationBtn.style.display = 'none';

      showStatus('Автоматизация остановлена', 'info');
      stopProgressPolling();
    } catch (error) {
      showStatus('Ошибка остановки: ' + error.message, 'warning');
      console.error(error);
    }
  });

  // Продолжение сохраненной автоматизации
  resumeAutomationBtn.addEventListener('click', async () => {
    try {
      const storage = await chrome.storage.local.get(['automationConfig', 'automationProgress']);

      if (!storage.automationConfig) {
        showStatus('Нет сохраненного прогресса', 'warning');
        return;
      }

      const config = {
        ...storage.automationConfig,
        currentDate: storage.automationProgress.currentDate,
        currentPage: storage.automationProgress.currentPage || 1
      };

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab || !tab.id) {
        showStatus('Не удалось найти активную вкладку', 'warning');
        return;
      }

      await chrome.runtime.sendMessage({
        type: 'START_AUTOMATION',
        config: config,
        tabId: tab.id
      });

      automationProgress.style.display = 'block';
      startAutomationBtn.style.display = 'none';
      stopAutomationBtn.style.display = 'block';
      resumeAutomationBtn.style.display = 'none';

      showStatus('Продолжение автоматизации', 'success');
      startProgressPolling();
    } catch (error) {
      showStatus('Ошибка продолжения: ' + error.message, 'warning');
      console.error(error);
    }
  });

  // Polling прогресса
  let progressInterval = null;

  function startProgressPolling() {
    // Очистить предыдущий интервал если был
    if (progressInterval) {
      clearInterval(progressInterval);
    }

    progressInterval = setInterval(async () => {
      try {
        const response = await chrome.runtime.sendMessage({
          type: 'GET_AUTOMATION_STATUS'
        });

        if (response.success && response.state.progress) {
          updateProgressUI(response.state.progress);
        }
      } catch (error) {
        console.error('[Popup] Error polling progress:', error);
        // Если ошибка - возможно автоматизация завершена
        stopProgressPolling();
      }
    }, 1000);
  }

  function stopProgressPolling() {
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
  }

  function updateProgressUI(progress) {
    const { config, stats } = progress;

    if (!config) return;

    // Обновить прогресс бар
    const percent = calculateProgress(config, stats);
    progressFill.style.width = `${percent}%`;

    // Обновить текст
    const dateInfo = config.currentDate || '-';
    const pageInfo = config.currentPage && config.totalPages
      ? `${config.currentPage}/${config.totalPages}`
      : '-';
    const datesInfo = stats ? stats.processedDates : 0;

    progressStatus.textContent = `Дата: ${dateInfo} | Страница: ${pageInfo} | Обработано дат: ${datesInfo}`;
  }

  function calculateProgress(config, stats) {
    if (!config.startDate || !config.endDate) return 0;

    // Получить все даты из select
    const startIndex = Array.from(startDateSelect.options).findIndex(
      opt => opt.value === config.startDate
    );
    const endIndex = Array.from(endDateSelect.options).findIndex(
      opt => opt.value === config.endDate
    );

    if (startIndex === -1 || endIndex === -1) return 0;

    const totalDates = endIndex - startIndex + 1;
    if (totalDates === 0) return 0;

    const processedDates = stats ? stats.processedDates : 0;
    const currentPageProgress = config.totalPages > 0
      ? (config.currentPage || 0) / config.totalPages
      : 0;

    return Math.round(((processedDates + currentPageProgress) / totalDates) * 100);
  }

  // Загружаем статистику и даты при открытии
  await loadStats();
  await loadAvailableDates();
  await checkSavedProgress();
});
