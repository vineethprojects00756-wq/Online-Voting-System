(() => {
    const forms = document.querySelectorAll('form.smart-form');
    forms.forEach(form => {
        form.addEventListener('submit', () => {
            form.classList.add('is-submitting');
        });
    });
})();
