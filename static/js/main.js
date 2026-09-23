/* TopicLens — Main JS */
document.addEventListener('DOMContentLoaded', function () {

    /* ── AOS init ─────────────────────────────────── */
    if (typeof AOS !== 'undefined') {
        AOS.init({ duration: 500, once: true, offset: 30 });
    }

    /* ── Sidebar toggle (mobile) ──────────────────── */
    const sidebar  = document.getElementById('sidebar');
    const toggle   = document.getElementById('sidebarToggle');
    const overlay  = document.getElementById('sidebarOverlay');

    function openSidebar()  { sidebar.classList.add('show'); overlay.classList.add('show'); }
    function closeSidebar() { sidebar.classList.remove('show'); overlay.classList.remove('show'); }

    if (toggle)  toggle.addEventListener('click', openSidebar);
    if (overlay) overlay.addEventListener('click', closeSidebar);

    /* ── Auto-dismiss flash messages ──────────────── */
    document.querySelectorAll('.flash-msg').forEach(function (msg) {
        setTimeout(function () {
            msg.style.transition = 'opacity .3s, transform .3s';
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-8px)';
            setTimeout(function () { msg.remove(); }, 300);
        }, 5000);
    });
});
