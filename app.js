document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const themeToggleBtn = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const form = document.getElementById('symptom-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingEl = document.getElementById('loading');
    const resultsContainer = document.getElementById('results');
    const historyList = document.getElementById('history-list');

    // State
    const API_URL = '/api/diagnose';
    let history = JSON.parse(localStorage.getItem('medi_assist_history')) || [];

    // Initialize Theme
    const initTheme = () => {
        const savedTheme = localStorage.getItem('medi_assist_theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.body.classList.replace('light-mode', 'dark-mode');
            themeIcon.classList.replace('ph-moon', 'ph-sun');
        }
    };

    // Toggle Theme
    themeToggleBtn.addEventListener('click', () => {
        const isDark = document.body.classList.contains('dark-mode');
        
        if (isDark) {
            document.body.classList.replace('dark-mode', 'light-mode');
            themeIcon.classList.replace('ph-sun', 'ph-moon');
            localStorage.setItem('medi_assist_theme', 'light');
        } else {
            document.body.classList.replace('light-mode', 'dark-mode');
            themeIcon.classList.replace('ph-moon', 'ph-sun');
            localStorage.setItem('medi_assist_theme', 'dark');
        }
    });

    // Render History
    const renderHistory = () => {
        historyList.innerHTML = '';
        
        if (history.length === 0) {
            historyList.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.9rem; text-align: center; margin-top: 2rem;">No previous consultations</p>';
            return;
        }

        // Sort by date descending
        const sortedHistory = [...history].sort((a, b) => new Date(b.date) - new Date(a.date));

        sortedHistory.forEach((item, index) => {
            const historyEl = document.createElement('div');
            historyEl.className = 'history-item';
            
            const dateObj = new Date(item.date);
            const dateStr = dateObj.toLocaleDateString() + ' ' + dateObj.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            // Truncate symptoms for snippet
            const snippet = item.symptoms.length > 30 ? item.symptoms.substring(0, 30) + '...' : item.symptoms;

            historyEl.innerHTML = `
                <span class="date">${dateStr}</span>
                <span class="snippet">"${snippet}"</span>
            `;
            
            historyEl.addEventListener('click', () => {
                displayResults(item.response);
                // Scroll to top on mobile
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });

            historyList.appendChild(historyEl);
        });
    };

    // Save to History
    const saveToHistory = (symptoms, response) => {
        const newItem = {
            id: Date.now().toString(),
            date: new Date().toISOString(),
            symptoms: symptoms,
            response: response
        };
        
        history.unshift(newItem); // Add to beginning
        // Keep only last 10 entries to manage storage limit
        if (history.length > 10) history = history.slice(0, 10);
        
        localStorage.setItem('medi_assist_history', JSON.stringify(history));
        renderHistory();
    };

    // Display Results in UI
    const displayResults = (data) => {
        resultsContainer.innerHTML = '';
        resultsContainer.classList.remove('hidden');

        // Create Sections
        const createSection = (title, items, iconClass, typeClass) => {
            if (!items || items.length === 0) return '';
            
            const listItems = items.map(item => `<li>${item}</li>`).join('');
            return `
                <div class="result-section ${typeClass}">
                    <h3><i class="ph-fill ${iconClass}"></i> ${title}</h3>
                    <ul class="result-list">
                        ${listItems}
                    </ul>
                </div>
            `;
        };

        const conditionsHtml = createSection('Possible Conditions', data.possible_conditions, 'ph-stethoscope', 'conditions-section');
        const remediesHtml = createSection('Home Remedies & Advice', data.remedies_and_advice, 'ph-leaf', 'remedies-section');
        
        let doctorHtml = '';
        if (data.doctor_recommendation) {
            doctorHtml = `
                <div class="result-section advice-section">
                    <h3><i class="ph-fill ph-user-md"></i> Medical Recommendation</h3>
                    <div class="doctor-recommendation">
                        ${data.doctor_recommendation}
                    </div>
                </div>
            `;
        }

        resultsContainer.innerHTML = conditionsHtml + remediesHtml + doctorHtml;
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    };

    // Handle Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const age = document.getElementById('age').value;
        const gender = document.getElementById('gender').value;
        const symptoms = document.getElementById('symptoms').value;

        if (!symptoms.trim()) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span>Analyzing...</span><div class="spinner" style="width: 20px; height: 20px; border-width: 2px;"></div>';
        resultsContainer.classList.add('hidden');
        loadingEl.classList.remove('hidden');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ age, gender, symptoms }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze symptoms.');
            }

            // Display and Save
            displayResults(data);
            saveToHistory(symptoms, data);

        } catch (error) {
            console.error('Error:', error);
            resultsContainer.innerHTML = `
                <div class="disclaimer-banner" style="background-color: var(--danger-bg); border-color: var(--danger-color); color: var(--danger-color);">
                    <i class="ph-fill ph-warning-circle"></i>
                    <p><strong>Error:</strong> ${error.message} <br>Ensure the backend is running (e.g. <code>python app.py</code> on port 5000) and open the app at <code>http://localhost:5000</code>. If using Gemini, set <code>GEMINI_API_KEY</code> in <code>.env</code>.</p>
                </div>
            `;
            resultsContainer.classList.remove('hidden');
        } finally {
            // Restore UI State
            loadingEl.classList.add('hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<span>Analyze Symptoms</span><i class="ph-bold ph-sparkle"></i>';
        }
    });

    // Run initializations
    initTheme();
    renderHistory();
});
