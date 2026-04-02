// Form validation and enhancement
document.addEventListener('DOMContentLoaded', function() {
    // Get form elements
    const searchForm = document.getElementById('searchForm');
    const submitBtn = document.getElementById('submitBtn');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const pdfForm = document.querySelector('form[action="/download_pdf"]');

    // Set default dates if empty
    const startDate = document.getElementById('start_date');
    const endDate = document.getElementById('end_date');
    
    if (startDate && !startDate.value) {
        const twoDaysAgo = new Date();
        twoDaysAgo.setDate(twoDaysAgo.getDate() - 2);
        startDate.value = twoDaysAgo.toISOString().split('T')[0];
    }
    
    if (endDate && !endDate.value) {
        const today = new Date();
        endDate.value = today.toISOString().split('T')[0];
    }

    // Form validation
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const districts = document.getElementById('districts').value.trim();

            if (!districts) {
                e.preventDefault();
                showError('Please enter at least one district');
                return;
            }

            // Show loading state
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = `
                    <span class="loading"></span>
                    <span>Searching...</span>
                `;
            }
        });
    }

    // PDF download with loading state
    if (pdfForm && downloadPdfBtn) {
        pdfForm.addEventListener('submit', function(e) {
            downloadPdfBtn.disabled = true;
            downloadPdfBtn.innerHTML = `
                <span class="loading"></span>
                <span>Generating PDF...</span>
            `;

            // Re-enable after 10 seconds (in case something goes wrong)
            setTimeout(() => {
                if (downloadPdfBtn) {
                    downloadPdfBtn.disabled = false;
                    downloadPdfBtn.innerHTML = `
                        <span class="btn-icon">📥</span>
                        <span>Download PDF</span>
                    `;
                }
            }, 10000);
        });
    }

    // Error message helper
    function showError(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'error';
        alertDiv.innerHTML = `
            <strong>⚠️ Error:</strong> ${message}
        `;
        
        const container = document.querySelector('.container');
        const searchCard = document.querySelector('.search-card');
        
        if (container && searchCard) {
            container.insertBefore(alertDiv, searchCard);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
    }

    // Input validation for districts (only letters, commas, spaces)
    const districtsInput = document.getElementById('districts');
    if (districtsInput) {
        districtsInput.addEventListener('input', function(e) {
            this.value = this.value.replace(/[^a-zA-Z,\s]/g, '');
        });
    }

    // Language codes validation
    const languagesInput = document.getElementById('languages');
    if (languagesInput) {
        languagesInput.addEventListener('input', function(e) {
            this.value = this.value.replace(/[^a-zA-Z,\s]/g, '').toLowerCase();
        });
    }

    // Date range validation
    if (startDate && endDate) {
        [startDate, endDate].forEach(dateInput => {
            dateInput.addEventListener('change', function() {
                if (startDate.value && endDate.value) {
                    if (new Date(startDate.value) > new Date(endDate.value)) {
                        showError('Start date cannot be after end date');
                        endDate.value = startDate.value;
                    }
                }
            });
        });
    }

    // Max articles validation
    const maxArticles = document.getElementById('max_articles');
    if (maxArticles) {
        maxArticles.addEventListener('input', function() {
            let value = parseInt(this.value);
            if (value < 1) this.value = 1;
            if (value > 100) this.value = 100;
        });
    }

    // Settings page - toggle NewsAPI key visibility
    const useNewsapiCheckbox = document.querySelector('input[name="use_newsapi"]');
    const newsapiKeyInput = document.getElementById('newsapi_key');
    
    if (useNewsapiCheckbox && newsapiKeyInput) {
        const toggleNewsapiKey = function() {
            newsapiKeyInput.disabled = !useNewsapiCheckbox.checked;
            const newsapiCard = useNewsapiCheckbox.closest('.source-card');
            if (newsapiCard) {
                newsapiCard.classList.toggle('source-card-active', useNewsapiCheckbox.checked);
            }
        };
        
        useNewsapiCheckbox.addEventListener('change', toggleNewsapiKey);
        toggleNewsapiKey();
    }

    // Smooth scroll to articles
    const articlesContainer = document.querySelector('.articles-container');
    if (articlesContainer) {
        articlesContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // Add copy functionality for article URLs
    document.querySelectorAll('.source-link').forEach(link => {
        link.addEventListener('click', function(e) {
            // Track clicks
            console.log('Article link clicked:', this.href);
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit form
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && searchForm) {
            e.preventDefault();
            searchForm.dispatchEvent(new Event('submit'));
        }
        
        // Escape to go back to dashboard
        if (e.key === 'Escape' && !document.getElementById('searchForm')) {
            window.location.href = '/';
        }
    });
});

// Add loading state for page navigation
window.addEventListener('beforeunload', function() {
    // Show loading indicator when leaving page
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-overlay';
    loadingDiv.innerHTML = '<div class="loading-spinner"></div>';
    document.body.appendChild(loadingDiv);
});
