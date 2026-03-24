(() => {
    const refreshButton = document.querySelector('[data-refresh-dashboard]');
    if (!refreshButton) {
        return;
    }

    refreshButton.addEventListener('click', () => {
        window.location.reload();
    });
})();
