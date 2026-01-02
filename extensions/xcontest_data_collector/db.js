// IndexedDB wrapper для хранения данных полетов
class FlightsDB {
  constructor() {
    this.dbName = 'XContestFlightsDB';
    this.version = 2; // Увеличили версию для пересоздания схемы
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error || new Error('Failed to open database'));
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const oldVersion = event.oldVersion;

        console.log('[FlightsDB] Upgrading database from version', oldVersion, 'to', this.version);

        // Если обновляемся с версии 1 на 2, удаляем старую схему
        if (oldVersion === 1 && db.objectStoreNames.contains('flights')) {
          console.log('[FlightsDB] Deleting old flights store');
          db.deleteObjectStore('flights');
        }

        // Store для полетов
        if (!db.objectStoreNames.contains('flights')) {
          console.log('[FlightsDB] Creating flights store');
          const flightsStore = db.createObjectStore('flights', { keyPath: 'id' });
          // ВАЖНО: ident НЕ уникальный, так как в XContest могут быть дубликаты
          flightsStore.createIndex('ident', 'ident', { unique: false });
          flightsStore.createIndex('timeClaim', 'timeClaim', { unique: false });
          flightsStore.createIndex('pilotId', 'pilot.id', { unique: false });
          flightsStore.createIndex('country', 'countries', { unique: false, multiEntry: true });
        }

        // Store для метаданных (статистика, настройки)
        if (!db.objectStoreNames.contains('metadata')) {
          console.log('[FlightsDB] Creating metadata store');
          db.createObjectStore('metadata', { keyPath: 'key' });
        }

        // Store для сырых API ответов (для отладки)
        if (!db.objectStoreNames.contains('apiResponses')) {
          console.log('[FlightsDB] Creating apiResponses store');
          const apiStore = db.createObjectStore('apiResponses', { keyPath: 'url' });
          apiStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
      };
    });
  }

  async addFlight(flight) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['flights'], 'readwrite');
      const store = transaction.objectStore('flights');
      const request = store.put(flight);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Failed to add flight'));
    });
  }

  async addFlights(flights) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['flights'], 'readwrite');
      const store = transaction.objectStore('flights');
      let addedCount = 0;
      let skippedCount = 0;
      let errors = [];
      let processedCount = 0;

      if (flights.length === 0) {
        resolve({ addedCount: 0, skippedCount: 0, total: 0 });
        return;
      }

      flights.forEach((flight, index) => {
        try {
          const request = store.put(flight);

          request.onsuccess = () => {
            addedCount++;
            processedCount++;
            console.log(`[FlightsDB] Added flight ${processedCount}/${flights.length}: ${flight?.id}`);
          };

          request.onerror = (event) => {
            processedCount++;
            const errorInfo = {
              index,
              flightId: flight?.id,
              flightIdent: flight?.ident,
              errorName: request.error?.name,
              errorMessage: request.error?.message,
              errorCode: request.error?.code
            };
            console.error('[FlightsDB] Error adding flight:', JSON.stringify(errorInfo, null, 2));
            console.error('[FlightsDB] Failed flight data:', flight);
            errors.push(errorInfo);

            // Дубль или ошибка
            if (request.error?.name === 'ConstraintError') {
              skippedCount++;
              console.log(`[FlightsDB] Skipped duplicate flight ${processedCount}/${flights.length}: ${flight?.id}`);
            } else {
              console.error(`[FlightsDB] Critical error on flight ${processedCount}/${flights.length}:`, request.error);
            }

            // Предотвращаем прерывание транзакции
            event.preventDefault();
          };
        } catch (error) {
          processedCount++;
          console.error('[FlightsDB] Exception while putting flight:', error, flight);
          errors.push({
            index,
            flightId: flight?.id,
            errorName: 'Exception',
            errorMessage: error?.message
          });
        }
      });

      transaction.oncomplete = () => {
        console.log('[FlightsDB] Transaction complete:', {
          total: flights.length,
          added: addedCount,
          skipped: skippedCount,
          processed: processedCount,
          errorsCount: errors.length
        });

        if (errors.length > 0 && errors.length < 10) {
          console.warn('[FlightsDB] Errors:', errors);
        } else if (errors.length >= 10) {
          console.warn('[FlightsDB] Many errors (showing first 10):', errors.slice(0, 10));
        }

        resolve({ addedCount, skippedCount, total: flights.length });
      };

      transaction.onerror = () => {
        console.error('[FlightsDB] Transaction error:', transaction.error);
        console.error('[FlightsDB] Transaction error name:', transaction.error?.name);
        console.error('[FlightsDB] Transaction error message:', transaction.error?.message);
        console.error('[FlightsDB] Processed:', processedCount, 'of', flights.length);
        console.error('[FlightsDB] Errors array:', JSON.stringify(errors, null, 2));

        const errorMessage = transaction.error?.message ||
                           (errors.length > 0 ? `Failed with ${errors.length} errors` : 'Failed to add flights');
        reject(new Error(errorMessage));
      };

      transaction.onabort = () => {
        console.error('[FlightsDB] Transaction aborted:', transaction.error);
        reject(transaction.error || new Error('Transaction aborted'));
      };
    });
  }

  async getFlight(id) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['flights'], 'readonly');
      const store = transaction.objectStore('flights');
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Failed to get flight'));
    });
  }

  async getAllFlights(limit = 1000) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['flights'], 'readonly');
      const store = transaction.objectStore('flights');
      const request = store.getAll(null, limit);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Failed to get flights'));
    });
  }

  async getFlightsCount() {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['flights'], 'readonly');
      const store = transaction.objectStore('flights');
      const request = store.count();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Failed to get count'));
    });
  }

  async clearFlights() {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['flights'], 'readwrite');
      const store = transaction.objectStore('flights');
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error || new Error('Failed to clear flights'));
    });
  }

  async saveApiResponse(url, data) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['apiResponses'], 'readwrite');
      const store = transaction.objectStore('apiResponses');
      const request = store.put({
        url,
        data,
        timestamp: new Date().toISOString()
      });

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Failed to save API response'));
    });
  }

  async getMetadata(key) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['metadata'], 'readonly');
      const store = transaction.objectStore('metadata');
      const request = store.get(key);

      request.onsuccess = () => resolve(request.result?.value);
      request.onerror = () => reject(request.error || new Error('Failed to get metadata'));
    });
  }

  async setMetadata(key, value) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction(['metadata'], 'readwrite');
      const store = transaction.objectStore('metadata');
      const request = store.put({ key, value });

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Failed to set metadata'));
    });
  }

  async exportToJSON() {
    const flights = await this.getAllFlights(100000); // Получить все полеты
    return JSON.stringify(flights, null, 2);
  }

  async getStats() {
    const count = await this.getFlightsCount();
    const lastUpdate = await this.getMetadata('lastUpdate');
    return { count, lastUpdate };
  }
}

// Экспорт для использования в других скриптах
if (typeof module !== 'undefined' && module.exports) {
  module.exports = FlightsDB;
}
