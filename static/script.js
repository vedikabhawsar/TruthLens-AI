
const newsInput = document.getElementById("newsInput");
const wordCount = document.getElementById("wordCount");

const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const sampleButton = document.getElementById("sampleButton");

const loadingMessage = document.getElementById("loadingMessage");
const errorMessage = document.getElementById("errorMessage");
const resultCard = document.getElementById("resultCard");

const predictionLabel = document.getElementById("predictionLabel");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const resultDisclaimer = document.getElementById("resultDisclaimer");


// Count words
function updateWordCount() {
    const text = newsInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;

    wordCount.textContent = `${words} words`;
}

newsInput.addEventListener("input", updateWordCount);


// Hide previous messages
function resetMessages() {
    errorMessage.hidden = true;
    loadingMessage.hidden = true;
    resultCard.hidden = true;
}


// Display error
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}


// Sample news text
sampleButton.addEventListener("click", () => {

    newsInput.value =
        "The local administration announced a new public transport " +
        "project on Monday. Officials said the project will improve " +
        "connectivity between residential areas, schools and major " +
        "business locations. The proposal will be reviewed by the " +
        "city council before implementation begins. A detailed public " +
        "report is expected to be released after the review process.";

    updateWordCount();
    resetMessages();
    newsInput.focus();
});


// Clear input and result
clearButton.addEventListener("click", () => {

    newsInput.value = "";

    updateWordCount();
    resetMessages();

    newsInput.focus();
});


// Analyze news
analyzeButton.addEventListener("click", async () => {

    const text = newsInput.value.trim();

    resetMessages();

    if (!text) {
        showError("Please enter news content before starting analysis.");
        return;
    }

    if (text.split(/\s+/).length < 20) {
        showError("Please enter at least 20 words for analysis.");
        return;
    }

    if (text.length > 10000) {
        showError("Please keep the content under 10,000 characters.");
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";
    loadingMessage.hidden = false;

    try {

        const response = await fetch("/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text
            })
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(
                result.message || "Something went wrong during analysis."
            );
        }

        // Display prediction
        predictionLabel.textContent = result.prediction;
        confidenceValue.textContent = `${result.confidence}%`;

        predictionLabel.classList.toggle(
            "fake",
            result.prediction === "FAKE"
        );

        confidenceBar.style.width = `${result.confidence}%`;

        resultDisclaimer.textContent =
            result.disclaimer +
            ` Text analyzed: ${result.word_count} words.`;

        resultCard.hidden = false;
        savePredictionHistory(result);

        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    } catch (error) {

        showError(
            error.message || "Unable to connect to the analysis server."
        );

    } finally {

        loadingMessage.hidden = true;
        analyzeButton.disabled = false;

        analyzeButton.innerHTML =
            'Analyze Content <span>→</span>';
    }
});


// Initial word count
updateWordCount();



// PREDICTION HISTORY

const HISTORY_KEY = "truthlens_prediction_history";

function getPredictionHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
        return [];
    }
}

function savePredictionHistory(result) {

    const history = getPredictionHistory();

    const entry = {
        prediction: result.prediction,
        confidence: result.confidence,
        date: new Date().toLocaleString()
    };

    history.unshift(entry);

    localStorage.setItem(
        HISTORY_KEY,
        JSON.stringify(history.slice(0, 5))
    );

    renderPredictionHistory();
}


function renderPredictionHistory() {

    const historyList = document.getElementById("historyList");

    if (!historyList) return;

    const history = getPredictionHistory();

    historyList.replaceChildren();

    if (history.length === 0) {
        const emptyMessage = document.createElement("p");
        emptyMessage.className = "empty-history";
        emptyMessage.textContent =
            "No predictions yet. Start analyzing news to build your history.";

        historyList.appendChild(emptyMessage);
        return;
    }

    history.forEach(item => {

        const row = document.createElement("div");
        row.className = "history-item";

        const details = document.createElement("div");

        const title = document.createElement("strong");
        title.textContent = item.prediction + " classification";

        const date = document.createElement("small");
        date.textContent = item.date;

        details.appendChild(title);
        details.appendChild(date);

        const badge = document.createElement("span");
        badge.className = "history-result";

        if (item.prediction === "FAKE") {
            badge.classList.add("fake");
        }

        badge.textContent =
            item.prediction + " · " + item.confidence + "%";

        row.appendChild(details);
        row.appendChild(badge);

        historyList.appendChild(row);
    });
}


// Clear saved history
const clearHistoryButton =
    document.getElementById("clearHistoryButton");

if (clearHistoryButton) {
    clearHistoryButton.addEventListener("click", () => {

        localStorage.removeItem(HISTORY_KEY);
        renderPredictionHistory();

    });
}


// Load history when website opens
renderPredictionHistory();

