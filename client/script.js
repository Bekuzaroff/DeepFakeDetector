// Конфигурация
const API_URL = 'http://localhost:8000/detect';

// DOM элементы
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const detectBtn = document.getElementById('detectBtn');
const loader = document.getElementById('loader');
const resultSection = document.getElementById('resultSection');
const resetBtn = document.getElementById('resetBtn');

// Состояние
let currentFile = null;

// Клик по области загрузки
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Drag & Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFileSelect(file);
    } else {
        showToast('Пожалуйста, загрузите изображение', 'error');
    }
});

// Выбор файла через input
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Обработка выбранного файла
function handleFileSelect(file) {
    // Проверка размера (макс 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showToast('Файл слишком большой (макс 10MB)', 'error');
        return;
    }
    
    currentFile = file;
    
    // Показываем превью
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        uploadArea.style.display = 'none';
        detectBtn.disabled = false;
        
        // Скрываем предыдущий результат
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Удалить выбранное фото
removeBtn.addEventListener('click', () => {
    resetUpload();
});

// Сброс загрузки
function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewSection.style.display = 'none';
    uploadArea.style.display = 'block';
    detectBtn.disabled = true;
    resultSection.style.display = 'none';
    previewImage.src = '';
}

// Определение дипфека
detectBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    // Показываем лоадер
    loader.classList.add('show');
    detectBtn.disabled = true;
    resultSection.style.display = 'none';
    
    // Отправляем запрос
    const formData = new FormData();
    formData.append('file', currentFile);
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка при анализе');
        }
        
        const data = await response.json();
        showResult(data);
        
    } catch (error) {
        console.error('Error:', error);
        showToast(error.message || 'Не удалось подключиться к серверу', 'error');
        detectBtn.disabled = false;
    } finally {
        loader.classList.remove('show');
    }
});

// Показать результат
function showResult(data) {
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultMessage = document.getElementById('resultMessage');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    if (data.is_fake) {
        resultIcon.textContent = '⚠️';
        resultTitle.textContent = 'Обнаружен дипфейк!';
        resultTitle.style.color = '#e74c3c';
        resultMessage.textContent = data.message;
        confidenceFill.style.backgroundColor = '#e74c3c';
    } else {
        resultIcon.textContent = '✅';
        resultTitle.textContent = 'Изображение реальное';
        resultTitle.style.color = '#27ae60';
        resultMessage.textContent = data.message;
        confidenceFill.style.backgroundColor = '#27ae60';
    }
    
    confidenceFill.style.width = `${data.confidence_percent}%`;
    confidenceText.textContent = `Уверенность: ${data.confidence_percent}%`;
    
    resultSection.style.display = 'block';
    detectBtn.disabled = false;
}

// Сброс и проверка другого фото
resetBtn.addEventListener('click', () => {
    resetUpload();
    resultSection.style.display = 'none';
});

// Toast уведомления
function showToast(message, type = 'info') {
    // Создаем toast если его нет
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        document.body.appendChild(toast);
        
        // Добавляем стили для toast
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            padding: 12px 24px;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            z-index: 1000;
            animation: slideUp 0.3s ease;
            max-width: 90%;
            text-align: center;
        `;
    }
    
    toast.style.backgroundColor = type === 'error' ? '#e74c3c' : '#3498db';
    toast.textContent = message;
    toast.style.display = 'block';
    
    setTimeout(() => {
        toast.style.display = 'none';
    }, 3000);
}

// Проверка соединения с сервером при загрузке
async function checkServer() {
    try {
        const response = await fetch('http://localhost:8000/');
        if (response.ok) {
            console.log('✅ Сервер запущен');
        }
    } catch (error) {
        console.warn('⚠️ Сервер не запущен, запустите backend: python main.py');
        showToast('Сервер не запущен. Запустите backend командой: python main.py', 'error');
    }
}

// Запускаем проверку при загрузке
checkServer();