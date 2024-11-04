// popup.js

document.addEventListener('DOMContentLoaded', () => {
    const statusElement = document.getElementById('adblocker-status');
    const toggleButton = document.getElementById('toggle-button');

    // Get the current status from storage
    chrome.storage.local.get(['isAdBlockerEnabled'], (result) => {
        const isEnabled = result.isAdBlockerEnabled || false;
        updateUI(isEnabled);
    });

    toggleButton.addEventListener('click', () => {
        chrome.storage.local.get(['isAdBlockerEnabled'], (result) => {
            const isEnabled = !(result.isAdBlockerEnabled || false);
            chrome.storage.local.set({ 'isAdBlockerEnabled': isEnabled }, () => {
                chrome.runtime.sendMessage({ action: 'toggleAdBlocker', enabled: isEnabled }, (response) => {
                    updateUI(isEnabled);
                });
            });
        });
    });

    function updateUI(isEnabled) {
        if (isEnabled) {
            statusElement.textContent = 'ON';
            statusElement.style.color = 'green';
            toggleButton.textContent = 'Disable AdBlocker';
        } else {
            statusElement.textContent = 'OFF';
            statusElement.style.color = 'red';
            toggleButton.textContent = 'Enable AdBlocker';
        }
    }
});
