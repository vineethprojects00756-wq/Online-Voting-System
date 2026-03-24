(() => {
    const buttons = document.querySelectorAll('[data-modal-close]');
    const modal = document.getElementById('global-modal');
    if (!modal) {
        return;
    }

    const closeModal = () => {
        modal.setAttribute('aria-hidden', 'true');
    };

    buttons.forEach(button => {
        button.addEventListener('click', closeModal);
    });
})();
