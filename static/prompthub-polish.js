/* Shared, progressively enhanced UI behaviour. */
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  document.querySelectorAll('.topnav a').forEach(link => {
    const active = link.classList.contains('active') || link.pathname === path || (link.pathname === '/descriptors' && /descriptor/.test(path));
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page');
  });
  const sidebar = document.querySelector('.sidebar-sections');
  if (sidebar) {
    const narrow = window.matchMedia('(max-width: 900px)');
    sidebar.open = !narrow.matches;
    narrow.addEventListener('change', event => sidebar.open = !event.matches);
  }

  const count = document.getElementById('selectionCount');
  const bulk = document.getElementById('bulkForm');
  if (count && bulk) {
    const updateSelection = () => {
      const selected = document.querySelectorAll('.bulkCheck:checked').length;
      count.textContent = `${selected} selected`;
      bulk.querySelectorAll('button[type="submit"]').forEach(button => button.disabled = !selected);
    };
    document.addEventListener('change', event => {
      if (event.target.matches('.bulkCheck')) updateSelection();
    });
    ['selectAll', 'selectNone'].forEach(id => document.getElementById(id)?.addEventListener('click', updateSelection));
    bulk.addEventListener('submit', event => {
      if (!document.querySelector('.bulkCheck:checked')) {
        event.preventDefault();
        window.showToast('Select at least one prompt first.');
      }
    });
    updateSelection();
  }
});
