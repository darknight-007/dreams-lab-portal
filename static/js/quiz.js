class QuizManager {
    constructor() {
        this.components = {};
        this.currentProgress = 0;
        this.totalQuestions = 0;
        this.init();
    }

    async init() {
        // Initialize quiz components
        document.querySelectorAll('.interactive-question').forEach(question => {
            const componentId = question.dataset.componentId;
            this.components[componentId] = {
                element: question,
                answered: false,
                correct: false
            };
            this.totalQuestions++;
        });

        // Load saved progress
        await this.loadProgress();

        // Set up event listeners
        this.setupEventListeners();
        this.updateProgress();
    }

    setupEventListeners() {
        // Handle answer submissions
        document.querySelectorAll('.answer-form').forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                const componentId = form.dataset.componentId;
                this.submitAnswer(componentId, form);
            });
        });

        // Handle hint requests
        document.querySelectorAll('.hint-button').forEach(button => {
            button.addEventListener('click', (e) => {
                const componentId = button.dataset.componentId;
                this.showHint(componentId);
            });
        });
    }

    async loadProgress() {
        try {
            const response = await fetch('/quiz/progress/load');
            const data = await response.json();
            
            // Update components with saved progress
            Object.entries(data).forEach(([componentId, progress]) => {
                if (this.components[componentId]) {
                    this.components[componentId].answered = true;
                    this.components[componentId].correct = progress.is_correct;
                    
                    // Update UI
                    const questionElement = document.querySelector(`[data-component-id="${componentId}"]`);
                    if (progress.is_correct) {
                        questionElement.classList.add('completed');
                        const feedbackPanel = questionElement.querySelector('.feedback-panel');
                        feedbackPanel.innerHTML = `
                            <div class="success">
                                <h4>Previously Completed ✓</h4>
                                <p>You've already solved this challenge correctly.</p>
                            </div>
                        `;
                    }
                }
            });
        } catch (error) {
            console.error('Error loading progress:', error);
        }
    }

    async saveProgress(componentId, isCorrect) {
        try {
            await fetch('/quiz/progress/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCsrfToken()
                },
                body: JSON.stringify({
                    component_id: componentId,
                    is_correct: isCorrect
                })
            });
        } catch (error) {
            console.error('Error saving progress:', error);
        }
    }

    async submitAnswer(componentId, form) {
        const answerInput = form.querySelector('input[name="answer"]');
        const feedbackPanel = form.closest('.interactive-question').querySelector('.feedback-panel');
        
        try {
            const response = await fetch(`/quiz/validate/${componentId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCsrfToken()
                },
                body: JSON.stringify({
                    answer: answerInput.value
                })
            });

            const data = await response.json();
            
            if (data.is_correct) {
                await this.handleCorrectAnswer(componentId, feedbackPanel, data);
            } else {
                this.handleIncorrectAnswer(componentId, feedbackPanel, data);
            }
            
            this.updateProgress();
            
        } catch (error) {
            console.error('Error submitting answer:', error);
            feedbackPanel.innerHTML = `<div class="error">An error occurred. Please try again.</div>`;
        }
    }

    async handleCorrectAnswer(componentId, feedbackPanel, data) {
        this.components[componentId].answered = true;
        this.components[componentId].correct = true;
        
        // Save progress
        await this.saveProgress(componentId, true);
        
        feedbackPanel.innerHTML = `
            <div class="success">
                <h4>Correct! 🎉</h4>
                <p>${data.message}</p>
                <div class="explanation">${data.explanation || ''}</div>
            </div>
        `;
        
        // Disable input and show next button if available
        const questionElement = feedbackPanel.closest('.interactive-question');
        questionElement.classList.add('completed');
        this.maybeShowNextQuestion(componentId);
    }

    handleIncorrectAnswer(componentId, feedbackPanel, data) {
        feedbackPanel.innerHTML = `
            <div class="error">
                <h4>Not quite right</h4>
                <p>${data.message}</p>
                ${data.hint ? `<div class="hint">${data.hint}</div>` : ''}
                <button class="try-again">Try Again</button>
            </div>
        `;
        
        feedbackPanel.querySelector('.try-again').addEventListener('click', () => {
            feedbackPanel.innerHTML = '';
        });
    }

    async showHint(componentId) {
        const hintPanel = document.querySelector(`#hint-panel-${componentId}`);
        
        try {
            const response = await fetch(`/quiz/hints/${componentId}`);
            const data = await response.json();
            
            if (data.hints && data.hints.length > 0) {
                const currentHintIndex = parseInt(hintPanel.dataset.currentHint) || 0;
                const hint = data.hints[currentHintIndex];
                
                hintPanel.innerHTML = `
                    <div class="hint-content">
                        <p>${hint}</p>
                        ${currentHintIndex < data.hints.length - 1 ? 
                            '<button class="next-hint">Next Hint</button>' : ''}
                    </div>
                `;
                
                hintPanel.dataset.currentHint = currentHintIndex + 1;
                
                const nextHintButton = hintPanel.querySelector('.next-hint');
                if (nextHintButton) {
                    nextHintButton.addEventListener('click', () => this.showHint(componentId));
                }
            }
        } catch (error) {
            console.error('Error fetching hints:', error);
            hintPanel.innerHTML = '<div class="error">Failed to load hint</div>';
        }
    }

    updateProgress() {
        const completed = Object.values(this.components).filter(c => c.correct).length;
        this.currentProgress = (completed / this.totalQuestions) * 100;
        
        const progressBar = document.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${this.currentProgress}%`;
            progressBar.setAttribute('aria-valuenow', this.currentProgress);
        }
        
        // Update progress text
        const progressText = document.querySelector('.progress-text');
        if (progressText) {
            progressText.textContent = `${completed} of ${this.totalQuestions} completed`;
        }
    }

    maybeShowNextQuestion(currentComponentId) {
        const currentQuestion = document.querySelector(`[data-component-id="${currentComponentId}"]`);
        const nextQuestion = currentQuestion.nextElementSibling;
        
        if (nextQuestion && nextQuestion.classList.contains('interactive-question')) {
            nextQuestion.scrollIntoView({ behavior: 'smooth' });
        } else if (this.currentProgress === 100) {
            this.showQuizComplete();
        }
    }

    showQuizComplete() {
        const completionMessage = document.createElement('div');
        completionMessage.className = 'quiz-complete';
        completionMessage.innerHTML = `
            <h3>Congratulations! 🎉</h3>
            <p>You've completed all the questions successfully!</p>
            <button onclick="location.reload()">Try Another Quiz</button>
        `;
        
        document.querySelector('.quiz-container').appendChild(completionMessage);
    }

    getCsrfToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]').value;
    }
}

// Initialize quiz when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.quizManager = new QuizManager();
}); 