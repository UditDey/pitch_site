// common.js - Shared functionality for all pages
// Loads: KaTeX (math), Prism (code highlighting), Tippy (tooltips)

(function () {

    // ===== KATEX (Math) =====
    const katexCSS = document.createElement('link');
    katexCSS.rel = 'stylesheet';
    katexCSS.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    document.head.appendChild(katexCSS);

    const katexJS = document.createElement('script');
    katexJS.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
    katexJS.onload = function () {
        const autoRender = document.createElement('script');
        autoRender.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js';
        autoRender.onload = function () {
            renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false }
                ]
            });
        };
        document.head.appendChild(autoRender);
    };
    document.head.appendChild(katexJS);

    // ===== PRISM (Code highlighting) =====
    const prismCSS = document.createElement('link');
    prismCSS.rel = 'stylesheet';
    prismCSS.href = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css';
    document.head.appendChild(prismCSS);

    const prismJS = document.createElement('script');
    prismJS.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js';
    prismJS.onload = function () {
        // Load language components
        const languages = ['python', 'verilog'];
        languages.forEach(lang => {
            const langScript = document.createElement('script');
            langScript.src = `https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-${lang}.min.js`;
            document.head.appendChild(langScript);
        });
    };
    document.head.appendChild(prismJS);

    // ===== TIPPY (Tooltips) =====
    const tippyCSS = document.createElement('link');
    tippyCSS.rel = 'stylesheet';
    tippyCSS.href = 'https://unpkg.com/tippy.js@6/animations/shift-away.css';
    document.head.appendChild(tippyCSS);

    const popper = document.createElement('script');
    popper.src = 'https://unpkg.com/@popperjs/core@2';
    popper.onload = function () {
        const tippy = document.createElement('script');
        tippy.src = 'https://unpkg.com/tippy.js@6';
        tippy.onload = initTooltips;
        document.head.appendChild(tippy);
    };
    document.head.appendChild(popper);

    function initTooltips() {
        // Auto-generate tooltips for external links only
        document.querySelectorAll('a[href^="http"]').forEach(link => {
            if (link.hasAttribute('data-tippy-content')) return;

            try {
                const url = new URL(link.href);
                link.setAttribute('data-tippy-content', url.hostname);
                link.classList.add('external');
            } catch (e) { }
        });

        // Initialize Tippy
        window.tippy('[data-tippy-content]', {
            animation: 'shift-away',
            arrow: true,
            theme: 'custom'
        });
    }
})();