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

      const result = await chrome.runtime.sendMessage({ type: 'EXPORT_DATA' });

      if (result.success) {
        // Создаем Blob и скачиваем
        const blob = new Blob([JSON.stringify(result.data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const dateStr = new Date().toISOString().split('T')[0];
        a.download = `xcontest_flights_${dateStr}.json`;
        a.click();
        URL.revokeObjectURL(url);

        showStatus(`Экспортировано ${result.count} полетов`, 'success');
      } else {
        showStatus('Ошибка экспорта', 'warning');
      }
    } catch (error) {
      console.error('[XContest Collector] Error exporting:', error);
      showStatus('Ошибка экспорта', 'warning');
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

  // Загружаем статистику при открытии
  await loadStats();
});
