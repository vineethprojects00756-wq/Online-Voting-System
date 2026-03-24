(() => {
    const voteForm = document.querySelector('form.smart-form');
    if (!voteForm) {
        return;
    }

    voteForm.addEventListener('submit', () => {
        const submitButton = voteForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Submitting...';
        }
    });
})();
