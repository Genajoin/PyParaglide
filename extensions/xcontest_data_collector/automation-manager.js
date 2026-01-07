// Менеджер автоматизации в content script
// Связывает popup/background с automation.js через custom events
(function() {
  'use strict';

  console.log('[XContest Automation Manager] Initializing');

  class AutomationManager {
    constructor() {
      this.setupListeners();
    }

    setupListeners() {
      // Слушаем прогресс от automation.js (инжектированного скрипта)
      window.addEventListener('xcontest-automation-progress', (event) => {
        console.log('[XContest Automation Manager] Progress update:', event.detail);
        this.sendToBackground({
          type: 'AUTOMATION_PROGRESS',
          data: event.detail
        });
      });

      // Слушаем завершение автоматизации
      window.addEventListener('xcontest-automation-complete', (event) => {
        console.log('[XContest Automation Manager] Automation complete:', event.detail);
        this.sendToBackground({
          type: 'AUTOMATION_COMPLETE',
          data: event.detail
        });
      });

      // Слушаем команды от content script через custom events
      // (content.js получает сообщения от popup и пересылает их сюда)
      window.addEventListener('xcontest-automation-command', (event) => {
        const { requestId, message } = event.detail;
        console.log('[XContest Automation Manager] Received command:', message.type, 'requestId:', requestId);

        this.handleCommand(message, (response) => {
          // Отправляем ответ обратно в content script
          window.dispatchEvent(new CustomEvent('xcontest-automation-response', {
            detail: {
              requestId: requestId,
              response: response
            }
          }));
        });
      });

      console.log('[XContest Automation Manager] Listeners set up');
    }

    /**
     * Обработка команд от popup/background
     * @param {Object} message - сообщение
     * @param {Function} sendResponse - функция для отправки ответа
     */
    handleCommand(message, sendResponse) {
      // Проверяем, что automation.js загружен
      const automation = window.xcontestAutomation;

      if (!automation && message.type !== 'GET_AVAILABLE_DATES') {
        sendResponse({ success: false, error: 'Automation not initialized' });
        return;
      }

      try {
        switch (message.type) {
          case 'START_AUTOMATION':
            console.log('[XContest Automation Manager] Starting automation with config:', message.config);
            automation.run(message.config)
              .then(() => sendResponse({ success: true }))
              .catch(err => sendResponse({ success: false, error: err.message }));
            break;

          case 'PAUSE_AUTOMATION':
            console.log('[XContest Automation Manager] Pausing automation');
            automation.pause();
            sendResponse({ success: true });
            break;

          case 'RESUME_AUTOMATION':
            console.log('[XContest Automation Manager] Resuming automation');
            automation.resume();
            sendResponse({ success: true });
            break;

          case 'STOP_AUTOMATION':
            console.log('[XContest Automation Manager] Stopping automation');
            automation.stop();
            sendResponse({ success: true });
            break;

          case 'GET_AVAILABLE_DATES':
            console.log('[XContest Automation Manager] Getting available dates');
            if (!automation) {
              sendResponse({ success: false, error: 'Automation not initialized' });
              break;
            }

            const dates = automation.getAvailableDates();
            sendResponse({ success: true, dates });
            break;

          case 'GET_AUTOMATION_STATE':
            console.log('[XContest Automation Manager] Getting automation state');
            if (!automation) {
              sendResponse({ success: false, error: 'Automation not initialized' });
              break;
            }

            const state = automation.getState();
            sendResponse({ success: true, state });
            break;

          default:
            console.warn('[XContest Automation Manager] Unknown command:', message.type);
            sendResponse({ success: false, error: 'Unknown command' });
        }
      } catch (error) {
        console.error('[XContest Automation Manager] Error handling command:', error);
        sendResponse({ success: false, error: error.message });
      }
    }

    /**
     * Отправка сообщения в background script через content.js
     * @param {Object} message - сообщение
     */
    sendToBackground(message) {
      // Отправляем событие в content.js, который перешлет его в background
      window.dispatchEvent(new CustomEvent('xcontest-to-background', {
        detail: message
      }));
    }
  }

  // Инициализация менеджера
  console.log('[XContest Automation Manager] Starting initialization...');
  console.log('[XContest Automation Manager] Document ready state:', document.readyState);
  console.log('[XContest Automation Manager] Checking window.xcontestAutomation:', typeof window.xcontestAutomation);

  // Проверяем, что automation.js уже загружен
  if (window.xcontestAutomation) {
    console.log('[XContest Automation Manager] ✓ Automation found, initializing manager');
    new AutomationManager();
  } else {
    console.error('[XContest Automation Manager] ✗ window.xcontestAutomation not found!');
    console.error('[XContest Automation Manager] automation.js should be loaded before automation-manager.js');
    console.error('[XContest Automation Manager] Please reload the page');
  }

})();
