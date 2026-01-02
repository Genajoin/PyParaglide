// Content script для перехвата API запросов XContest
(function() {
  'use strict';

  console.log('[XContest Collector] Content script loaded on:', window.location.href);

  // Инжектируем скрипт для перехвата fetch И XMLHttpRequest
  const script = document.createElement('script');
  script.src = chrome.runtime.getURL('injected.js');
  script.onload = function() {
    console.log('[XContest Collector] Injection script loaded');
    this.remove();
  };
  script.onerror = function() {
    console.error('[XContest Collector] Failed to load injection script');
  };

  // Вставляем скрипт на страницу как можно раньше
  (document.head || document.documentElement).appendChild(script);

  // Проверка доступности расширения
  function isExtensionContextValid() {
    try {
      return !!chrome.runtime?.id;
    } catch {
      return false;
    }
  }

  // Слушаем custom events от инжектированного скрипта
  window.addEventListener('xcontest-api-response', async (event) => {
    const { url, data } = event.detail;
    console.log('[XContest Collector] Received API data:', { url, itemsCount: data?.items?.length });

    // Проверяем, что расширение всё ещё активно
    if (!isExtensionContextValid()) {
      console.warn('[XContest Collector] Extension context invalidated - please reload the page');
      showReloadNotification();
      return;
    }

    // Отправляем данные в background script через message
    try {
      const response = await chrome.runtime.sendMessage({
        type: 'API_RESPONSE',
        url: url,
        data: data
      });

      console.log('[XContest Collector] Background response:', response);

      // Показываем уведомление пользователю
      if (response?.success && response?.stats) {
        showNotification(response.stats);
      }
    } catch (error) {
      // Специальная обработка ошибки инвалидированного контекста
      if (error.message && error.message.includes('Extension context invalidated')) {
        console.warn('[XContest Collector] Extension was reloaded - please refresh this page');
        showReloadNotification();
      } else {
        console.error('[XContest Collector] Error sending to background:', error);
      }
    }
  });

  // Показываем уведомление о сохранении
  function showNotification(stats) {
    // Создаем уведомление в правом верхнем углу
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 70px;
      right: 20px;
      background: #4CAF50;
      color: white;
      padding: 15px 20px;
      border-radius: 5px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.3);
      z-index: 10000;
      font-family: Arial, sans-serif;
      font-size: 14px;
      max-width: 300px;
    `;

    notification.innerHTML = `
      <strong>✓ Данные сохранены</strong><br/>
      Добавлено: ${stats.addedCount}<br/>
      Дубли: ${stats.skippedCount}<br/>
      Всего в БД: ${stats.totalInDB}
    `;

    document.body.appendChild(notification);

    // Удаляем уведомление через 4 секунды
    setTimeout(() => {
      notification.style.transition = 'opacity 0.5s';
      notification.style.opacity = '0';
      setTimeout(() => notification.remove(), 500);
    }, 4000);
  }

  // Показываем уведомление о необходимости перезагрузки
  function showReloadNotification() {
    // Проверяем, не показано ли уже уведомление
    if (document.getElementById('xcontest-reload-notification')) {
      return;
    }

    const notification = document.createElement('div');
    notification.id = 'xcontest-reload-notification';
    notification.style.cssText = `
      position: fixed;
      top: 70px;
      right: 20px;
      background: #ff9800;
      color: white;
      padding: 15px 20px;
      border-radius: 5px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.3);
      z-index: 10001;
      font-family: Arial, sans-serif;
      font-size: 14px;
      max-width: 300px;
      cursor: pointer;
    `;

    notification.innerHTML = `
      <strong>⚠ Расширение обновлено</strong><br/>
      Нажмите здесь, чтобы перезагрузить страницу
    `;

    notification.addEventListener('click', () => {
      window.location.reload();
    });

    document.body.appendChild(notification);
  }

  console.log('[XContest Collector] Content script initialized');
})();
