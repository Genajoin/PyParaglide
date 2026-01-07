// Автоматизация сбора данных с XContest.org
// Этот скрипт инжектируется в контекст страницы для доступа к window.ZenList
(function() {
  'use strict';

  console.log('[XContest Automation] Script starting execution...');
  console.log('[XContest Automation] Script URL:', document.currentScript?.src || 'inline');
  console.log('[XContest Automation] Document ready state:', document.readyState);

  class XContestAutomation {
    constructor() {
      this.state = 'stopped'; // stopped | running | paused
      this.config = null;     // { startDate, endDate, currentDate, currentPage, totalPages }
      this.stats = { totalFlights: 0, processedDates: 0, errors: [] };
    }

    /**
     * Получить список доступных дат из select элемента
     * @returns {Array} массив объектов {value, text, count}
     */
    getAvailableDates() {
      const dateSelect = document.querySelector('select[name="filter[date]"]');
      if (!dateSelect) {
        console.error('[XContest Automation] Date select not found - page may not be fully loaded');
        console.error('[XContest Automation] Current URL:', window.location.href);
        console.error('[XContest Automation] Try waiting a few seconds or reload the page');
        return [];
      }

      const dates = Array.from(dateSelect.options).map(opt => ({
        value: opt.value,  // YYYY-MM-DD
        text: opt.text,    // DD.MM.YYYY [COUNT]
        count: parseInt(opt.text.match(/\[(\d+)\]/)?.[1] || '0')
      }));

      console.log(`[XContest Automation] Found ${dates.length} available dates`);
      return dates;
    }

    /**
     * Сменить дату через DOM манипуляцию
     * @param {string} newDate - дата в формате YYYY-MM-DD
     */
    async changeDate(newDate) {
      console.log(`[XContest Automation] Changing date to: ${newDate}`);

      const dateSelect = document.querySelector('select[name="filter[date]"]');
      if (!dateSelect) {
        throw new Error('Date select not found');
      }

      dateSelect.value = newDate;
      dateSelect.dispatchEvent(new Event('change', { bubbles: true }));

      // Ждем загрузки данных
      await this.waitForDataLoad();
    }

    /**
     * Определить количество страниц из пагинатора
     * @returns {number} количество страниц
     */
    getTotalPages() {
      const pager = document.querySelector('div.XCpager');
      if (!pager) {
        console.log('[XContest Automation] No pager found - assuming 1 page');
        return 1;
      }

      const pageLinks = pager.querySelectorAll('a[href*="flights[start]"]');
      if (pageLinks.length === 0) {
        console.log('[XContest Automation] No page links found - assuming 1 page');
        return 1;
      }

      // Находим последнюю ссылку пагинации
      const lastPageLink = pageLinks[pageLinks.length - 1];
      const lastStart = parseInt(lastPageLink.href.match(/flights\[start\]=(\d+)/)?.[1] || '0');
      const totalPages = Math.ceil((lastStart + 100) / 100);

      console.log(`[XContest Automation] Found ${totalPages} pages (last start: ${lastStart})`);
      return totalPages;
    }

    /**
     * Перейти на следующую страницу
     * @param {number} currentPage - текущая страница (0-indexed)
     * @returns {boolean} успешность перехода
     */
    async goToNextPage(currentPage) {
      const pager = document.querySelector('div.XCpager');
      if (!pager) {
        console.error('[XContest Automation] Pager not found');
        return false;
      }

      const nextStart = currentPage * 100;
      const nextPageLink = pager.querySelector(`a[href*="flights[start]=${nextStart}"]`);

      if (!nextPageLink) {
        console.error(`[XContest Automation] Page link not found for start=${nextStart}`);
        return false;
      }

      if (!window.ZenList) {
        console.error('[XContest Automation] ZenList not found in window');
        return false;
      }

      console.log(`[XContest Automation] Navigating to page ${currentPage + 1} (start=${nextStart})`);

      // Вызываем ZenList.onclick для перехода на страницу
      window.ZenList.onclick(nextPageLink, ['flights']);

      // Ждем загрузки данных
      await this.waitForDataLoad();
      return true;
    }

    /**
     * Ожидание загрузки данных через прослушивание custom event
     * @param {number} timeout - максимальное время ожидания в мс
     * @returns {Promise}
     */
    waitForDataLoad(timeout = 5000) {
      return new Promise((resolve) => {
        let resolved = false;

        const handler = () => {
          if (!resolved) {
            resolved = true;
            window.removeEventListener('xcontest-api-response', handler);
            console.log('[XContest Automation] Data loaded (API response received)');
            resolve();
          }
        };

        window.addEventListener('xcontest-api-response', handler);

        // Таймаут на случай, если событие не придет
        setTimeout(() => {
          if (!resolved) {
            resolved = true;
            window.removeEventListener('xcontest-api-response', handler);
            console.warn('[XContest Automation] Data load timeout (no API response)');
            resolve();
          }
        }, timeout);
      });
    }

    /**
     * Основной цикл автоматизации
     * @param {Object} config - конфигурация {startDate, endDate, currentDate?, currentPage?}
     */
    async run(config) {
      console.log('[XContest Automation] Starting automation with config:', config);

      this.state = 'running';
      this.config = { ...config };

      // Получить диапазон дат для обработки
      const dates = this.getDateRange(config.startDate, config.endDate);

      if (dates.length === 0) {
        console.error('[XContest Automation] No dates to process');
        this.state = 'stopped';
        this.sendComplete();
        return;
      }

      console.log(`[XContest Automation] Processing ${dates.length} dates`);

      // Если указана currentDate - начинаем с неё (возобновление)
      const startIndex = config.currentDate
        ? dates.indexOf(config.currentDate)
        : 0;

      for (let i = startIndex; i < dates.length; i++) {
        if (this.state !== 'running') {
          console.log('[XContest Automation] Stopped by user');
          break;
        }

        const date = dates[i];
        this.config.currentDate = date;
        this.sendProgress();

        console.log(`[XContest Automation] Processing date ${i + 1}/${dates.length}: ${date}`);

        // Сменить дату
        await this.changeDate(date);

        // Определить количество страниц
        const totalPages = this.getTotalPages();
        this.config.totalPages = totalPages;

        // Определить начальную страницу (для возобновления)
        const startPage = (config.currentDate === date && config.currentPage)
          ? config.currentPage - 1
          : 0;

        // Обработать все страницы для текущей даты
        for (let page = startPage; page < totalPages; page++) {
          if (this.state !== 'running') {
            console.log('[XContest Automation] Stopped by user');
            break;
          }

          this.config.currentPage = page + 1;
          this.sendProgress();

          console.log(`[XContest Automation] Processing page ${page + 1}/${totalPages}`);

          // Если не первая страница - переходим
          if (page > 0) {
            const success = await this.goToNextPage(page);
            if (!success) {
              this.logError(`Failed to navigate to page ${page + 1} for date ${date}`);
              break; // Переходим к следующей дате
            }
          }

          // Задержка между страницами
          await this.delay(1000);
        }

        this.stats.processedDates++;

        // Задержка между датами
        if (i < dates.length - 1) {
          await this.delay(2000);
        }
      }

      console.log('[XContest Automation] Automation complete');
      this.state = 'stopped';
      this.sendComplete();
    }

    /**
     * Получить диапазон дат из доступных
     * @param {string} startDate - начальная дата
     * @param {string} endDate - конечная дата
     * @returns {Array} массив дат в формате YYYY-MM-DD
     */
    getDateRange(startDate, endDate) {
      const available = this.getAvailableDates();
      const startIdx = available.findIndex(d => d.value === startDate);
      const endIdx = available.findIndex(d => d.value === endDate);

      if (startIdx === -1) {
        console.error(`[XContest Automation] Start date not found: ${startDate}`);
        return [];
      }

      if (endIdx === -1) {
        console.error(`[XContest Automation] End date not found: ${endDate}`);
        return [];
      }

      if (startIdx > endIdx) {
        console.error('[XContest Automation] Start date is after end date');
        return [];
      }

      return available.slice(startIdx, endIdx + 1).map(d => d.value);
    }

    /**
     * Отправка прогресса в content script
     */
    sendProgress() {
      window.dispatchEvent(new CustomEvent('xcontest-automation-progress', {
        detail: {
          state: this.state,
          config: this.config,
          stats: this.stats
        }
      }));
    }

    /**
     * Сигнал о завершении
     */
    sendComplete() {
      window.dispatchEvent(new CustomEvent('xcontest-automation-complete', {
        detail: this.stats
      }));
    }

    /**
     * Задержка
     * @param {number} ms - миллисекунды
     */
    delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Логирование ошибки
     * @param {string} message - сообщение об ошибке
     */
    logError(message) {
      console.error(`[XContest Automation] Error: ${message}`);
      this.stats.errors.push({
        message,
        timestamp: new Date().toISOString()
      });
    }

    /**
     * Управление состоянием
     */
    pause() {
      console.log('[XContest Automation] Paused');
      this.state = 'paused';
    }

    resume() {
      if (this.state === 'paused') {
        console.log('[XContest Automation] Resumed');
        this.state = 'running';
      }
    }

    stop() {
      console.log('[XContest Automation] Stopped');
      this.state = 'stopped';
    }

    /**
     * Получить текущее состояние
     */
    getState() {
      return {
        state: this.state,
        config: this.config,
        stats: this.stats
      };
    }
  }

  // Создаем глобальный экземпляр
  try {
    console.log('[XContest Automation] Creating automation instance...');
    window.xcontestAutomation = new XContestAutomation();
    console.log('[XContest Automation] ✓ Automation instance created successfully');
    console.log('[XContest Automation] ✓ window.xcontestAutomation is available');
  } catch (error) {
    console.error('[XContest Automation] ✗ Failed to create automation instance:', error);
    console.error('[XContest Automation] Error stack:', error.stack);
  }

})();
