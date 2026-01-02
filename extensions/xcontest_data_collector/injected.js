// Этот скрипт инжектируется в контекст страницы для перехвата fetch/XHR
(function() {
  'use strict';

  console.log('[XContest Collector] Injected script running');

  // Перехват fetch
  const originalFetch = window.fetch;
  window.fetch = function(...args) {
    const url = args[0];

    // Логируем ВСЕ fetch запросы для отладки
    console.log('[XContest Collector] Fetch request:', url);

    // Проверяем API запросы
    if (typeof url === 'string' && url.includes('/api/data/')) {
      console.log('[XContest Collector] ✓ Intercepted API request:', url);

      return originalFetch.apply(this, args).then(response => {
        const clonedResponse = response.clone();

        clonedResponse.json().then(data => {
          console.log('[XContest Collector] ✓ API Response received:', {
            url: url,
            itemsCount: data?.items?.length,
            listInfo: data?.list
          });

          window.dispatchEvent(new CustomEvent('xcontest-api-response', {
            detail: { url: url, data: data }
          }));
        }).catch(err => {
          console.error('[XContest Collector] ✗ Error reading JSON:', err);
        });

        return response;
      });
    }

    return originalFetch.apply(this, args);
  };

  // Перехват XMLHttpRequest (на случай если сайт использует XHR)
  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function(method, url, ...rest) {
    this._url = url;
    console.log('[XContest Collector] XHR open:', method, url);
    return originalOpen.apply(this, [method, url, ...rest]);
  };

  XMLHttpRequest.prototype.send = function(...args) {
    if (this._url && this._url.includes('/api/data/')) {
      console.log('[XContest Collector] ✓ Intercepted XHR API request:', this._url);

      this.addEventListener('load', function() {
        try {
          const data = JSON.parse(this.responseText);
          console.log('[XContest Collector] ✓ XHR API Response:', {
            url: this._url,
            itemsCount: data?.items?.length
          });

          window.dispatchEvent(new CustomEvent('xcontest-api-response', {
            detail: { url: this._url, data: data }
          }));
        } catch (err) {
          console.error('[XContest Collector] ✗ Error parsing XHR response:', err);
        }
      });
    }

    return originalSend.apply(this, args);
  };

  console.log('[XContest Collector] Interception hooks installed');
})();
