// Content script для перехвата API запросов XContest
(function() {
  'use strict';

  console.log('[XContest Collector] Content script loaded on:', window.location.href);

  // Функция для инжектирования скриптов
  function injectScript(src, name) {
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = chrome.runtime.getURL(src);
      script.onload = function() {
        console.log(`[XContest Collector] ${name} loaded`);
        // НЕ удаляем скрипт сразу - даем время на выполнение
        // Удалим через небольшую задержку
        setTimeout(() => {
          this.remove();
          console.log(`[XContest Collector] ${name} removed from DOM`);
        }, 500);
        resolve();
      };
      script.onerror = function() {
        console.error(`[XContest Collector] Failed to load ${name}`);
        reject(new Error(`Failed to load ${name}`));
      };

      // Вставляем скрипт
      (document.head || document.documentElement).appendChild(script);
    });
  }

  // Функция для инжектирования всех скриптов
  async function injectAllScripts() {
    try {
      // Сначала injected.js для перехвата API
      await injectScript('injected.js', 'injected.js');

      // Затем automation.js для автоматизации
      await injectScript('automation.js', 'automation.js');

      // Даем время скрипту automation.js выполниться
      await new Promise(resolve => setTimeout(resolve, 100));

      // И наконец automation-manager.js для управления
      // ВАЖНО: Он должен быть в контексте страницы, чтобы иметь доступ к window.xcontestAutomation
      await injectScript('automation-manager.js', 'automation-manager.js');

      console.log('[XContest Collector] All scripts injected successfully');
    } catch (error) {
      console.error('[XContest Collector] Error injecting scripts:', error);
    }
  }

  // Ждем готовности DOM перед инжектированием
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectAllScripts);
  } else {
    // DOM уже готов
    injectAllScripts();
  }

  // Проверка доступности расширения
  function isExtensionContextValid() {
    try {
      return !!chrome.runtime?.id;
    } catch {
      return false;
    }
  }

  // === Мост между popup и automation-manager (в контексте страницы) ===

  // Слушаем команды от popup/background через chrome.runtime.onMessage
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log('[XContest Collector] Received message from popup/background:', message.type);

    // Команды для автоматизации - отправляем в контекст страницы через custom event
    if (message.type === 'GET_AVAILABLE_DATES' ||
        message.type === 'START_AUTOMATION' ||
        message.type === 'STOP_AUTOMATION' ||
        message.type === 'PAUSE_AUTOMATION' ||
        message.type === 'RESUME_AUTOMATION' ||
        message.type === 'GET_AUTOMATION_STATE') {

      // Создаем уникальный ID для отслеживания ответа
      const requestId = Math.random().toString(36).substring(7);

      // Слушаем ответ от контекста страницы
      const responseHandler = (event) => {
        if (event.detail.requestId === requestId) {
          window.removeEventListener('xcontest-automation-response', responseHandler);
          console.log('[XContest Collector] Received response from page context:', event.detail);
          sendResponse(event.detail.response);
        }
      };

      window.addEventListener('xcontest-automation-response', responseHandler);

      // Отправляем команду в контекст страницы
      window.dispatchEvent(new CustomEvent('xcontest-automation-command', {
        detail: {
          requestId: requestId,
          message: message
        }
      }));

      return true; // Асинхронный ответ
    }

    // Остальные сообщения игнорируем (обрабатываются в другом месте)
    return false;
  });

  // Слушаем сообщения от automation-manager для отправки в background
  window.addEventListener('xcontest-to-background', async (event) => {
    const message = event.detail;
    console.log('[XContest Collector] Forwarding message to background:', message.type);

    // Проверяем, что расширение всё ещё активно
    if (!isExtensionContextValid()) {
      console.warn('[XContest Collector] Extension context invalidated - cannot forward message');
      return;
    }

    try {
      await chrome.runtime.sendMessage(message);
    } catch (error) {
      console.error('[XContest Collector] Error forwarding to background:', error);
    }
  });

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
