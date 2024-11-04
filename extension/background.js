// background.js

let isAdBlockerEnabled = false;

function updateProxySettings() {
    if (isAdBlockerEnabled) {
        // Enable proxy
        chrome.proxy.settings.set(
            {
                value: {
                    mode: "fixed_servers",
                    rules: {
                        singleProxy: {
                            scheme: "http",
                            host: "localhost",
                            port: 8080
                        },
                        bypassList: ["localhost"]
                    }
                },
                scope: 'regular'
            },
            () => {}
        );
    } else {
        // Disable proxy
        chrome.proxy.settings.set(
            {
                value: {
                    mode: "system"
                },
                scope: 'regular'
            },
            () => {}
        );
    }
}

// Initialize the state when the service worker starts
chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.get(['isAdBlockerEnabled'], (result) => {
        isAdBlockerEnabled = result.isAdBlockerEnabled || false;
        updateProxySettings();
    });
});

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'toggleAdBlocker') {
        isAdBlockerEnabled = request.enabled;
        updateProxySettings();
        sendResponse({ status: 'ok' });
    }
    return true; // Indicates you wish to send a response asynchronously
});
