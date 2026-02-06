// common.js - Shared functionality for all pages
// Loads: KaTeX (math), Prism (code highlighting), Tippy (tooltips)

(function () {
    // Helpers
    function loadCSS(href) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = href;
        document.head.appendChild(link);
    }

    function loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    // ===== KATEX (Math) =====
    async function initKaTeX() {
        loadCSS('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css');
        await loadScript('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js');
        await loadScript('https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js');
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
            ],
            strict: false
        });
    }

    // ===== PRISM (Code highlighting) =====
    async function initPrism() {
        loadCSS('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css');

        // Disable auto-highlight - we'll trigger it manually after languages load
        window.Prism = window.Prism || {};
        window.Prism.manual = true;

        await loadScript('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js');

        // Load language components
        await Promise.all([
            loadScript('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js'),
            loadScript('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-verilog.min.js')
        ]);

        // cpp depends on c, so load sequentially
        await loadScript('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-c.min.js');
        await loadScript('https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-cpp.min.js');

        // Inject our overrides AFTER Prism's CSS is loaded
        const overrides = document.createElement('style');
        overrides.textContent = `
            code[class*="language-"],
            pre[class*="language-"] {
                font-size: 0.8rem;
                line-height: 1.5;
            }
        `;
        document.head.appendChild(overrides);

        Prism.highlightAll();
    }

    // ===== TIPPY (Tooltips) =====
    async function initTippy() {
        loadCSS('https://unpkg.com/tippy.js@6/animations/shift-away.css');
        await loadScript('https://unpkg.com/@popperjs/core@2');
        await loadScript('https://unpkg.com/tippy.js@6');

        // Auto-generate tooltips for external links
        document.querySelectorAll('a[href^="http"]').forEach(link => {
            if (link.hasAttribute('data-tippy-content')) return;
            try {
                const url = new URL(link.href);
                link.setAttribute('data-tippy-content', url.hostname);
                link.classList.add('external');
            } catch (e) { }
        });

        tippy('[data-tippy-content]', {
            animation: 'shift-away',
            arrow: true,
            theme: 'custom'
        });
    }

    // Run all in parallel (they don't depend on each other)
    Promise.all([initKaTeX(), initPrism(), initTippy()]);
})();