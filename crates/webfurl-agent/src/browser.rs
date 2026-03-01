use std::sync::Arc;

use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use chromiumoxide::page::ScreenshotParams;
use chromiumoxide::Page;
use futures::StreamExt;
use tokio::task::JoinHandle;
use tracing::{debug, error, info};

/// JS snippet to detect loading indicators on the page (spinners, skeletons, "Loading" text).
/// Returns true if the page appears to still be loading.
const DETECT_LOADING_JS: &str = r#"
    (() => {
        // Check for common loading/spinner elements
        const selectors = [
            '[class*="loading" i]', '[class*="spinner" i]', '[class*="skeleton" i]',
            '[class*="shimmer" i]', '[class*="placeholder" i]',
            '[role="progressbar"]', '[aria-busy="true"]',
            '.loading', '.spinner', '.skeleton',
        ];
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.offsetParent !== null) return true;
        }
        // Check for visible text that says "Loading"
        const body = document.body ? document.body.innerText : '';
        if (/^Loading\.{0,3}$/m.test(body.trim())) return true;
        // Check if there are very few visible elements (page hasn't rendered yet)
        const visible = document.querySelectorAll('a, button, img, input, h1, h2, h3, p, li, td');
        if (visible.length < 3 && document.body && document.body.innerHTML.length > 500) return true;
        return false;
    })()
"#;

/// JS snippet that syncs runtime .value into DOM attributes so they appear in .content().
const SYNC_INPUT_VALUES_JS: &str = r#"
    document.querySelectorAll('input, textarea, select').forEach(el => {
        if (el.tagName === 'SELECT') {
            el.querySelectorAll('option').forEach(opt => {
                if (opt.selected) opt.setAttribute('selected', 'selected');
                else opt.removeAttribute('selected');
            });
        } else if (el.type === 'checkbox' || el.type === 'radio') {
            if (el.checked) el.setAttribute('checked', 'checked');
            else el.removeAttribute('checked');
        } else {
            el.setAttribute('value', el.value);
        }
    });
"#;

/// Manages a Chrome browser instance via CDP for page navigation and interaction.
pub struct BrowserSession {
    browser: Browser,
    _handler_task: JoinHandle<()>,
    current_page: Option<Arc<Page>>,
}

impl BrowserSession {
    pub async fn launch() -> Result<Self, String> {
        info!("launching Chrome browser...");

        let headless = std::env::var("WEBFURL_HEADLESS").map(|v| v == "1" || v == "true").unwrap_or(false);
        if headless {
            info!("running Chrome in headless mode (WEBFURL_HEADLESS=1)");
        } else {
            info!("running Chrome in visible mode (set WEBFURL_HEADLESS=1 to hide)");
        }

        let mut builder = BrowserConfig::builder()
            .disable_default_args()
            .viewport(None)
            .arg("--no-first-run")
            .arg("--no-default-browser-check")
            .arg("--disable-popup-blocking")
            .arg("--disable-prompt-on-repost")
            .arg("--disable-sync")
            .arg("--no-sandbox")
            .arg("--lang=en_US")
            .arg("--window-size=1920,1080")
            .arg("--remote-debugging-port=0");

        if headless {
            builder = builder.new_headless_mode();
        } else {
            builder = builder.with_head();
        }

        if let Ok(path) = std::env::var("CHROME_PATH") {
            info!(path = %path, "using Chrome from CHROME_PATH env");
            builder = builder.chrome_executable(path);
        } else {
            let candidates = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/snap/chromium/current/usr/lib/chromium-browser/chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
                "/usr/bin/chromium",
            ];
            for candidate in &candidates {
                if std::path::Path::new(candidate).exists() {
                    info!(path = %candidate, "found Chrome/Chromium");
                    builder = builder.chrome_executable(candidate);
                    break;
                }
            }
        }

        let config = builder.build()
            .map_err(|e| format!("Failed to build browser config: {e}"))?;

        let (browser, mut handler) = Browser::launch(config)
            .await
            .map_err(|e| format!("Failed to launch Chrome: {e}"))?;

        let handler_task = tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if let Err(e) = event {
                    let msg = e.to_string();
                    if msg.contains("did not match any variant of untagged enum") {
                        continue;
                    }
                    error!("CDP handler error: {e}");
                }
            }
        });

        info!("Chrome browser launched");

        let mut session = Self {
            browser,
            _handler_task: handler_task,
            current_page: None,
        };

        if let Ok(page) = session.browser.new_page("about:blank").await {
            session.current_page = Some(Arc::new(page));
        }

        Ok(session)
    }

    /// Navigate to a URL. Returns the rendered HTML.
    /// Uses JS navigation (fire-and-forget) so the browser renders live.
    pub async fn navigate(&mut self, url: &str) -> Result<String, String> {
        info!(url, "navigating browser to URL");

        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let escaped_url = url.replace('\\', "\\\\").replace('\'', "\\'");
        let nav_js = format!("window.location.assign('{escaped_url}')");
        let _ = page.evaluate(nav_js).await;

        // Phase 1: Wait for document.readyState === "complete"
        let mut attempts = 0;
        let mut saw_loading = false;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            attempts += 1;

            match page.evaluate("document.readyState").await {
                Ok(result) => {
                    if let Ok(state) = result.into_value::<String>() {
                        match state.as_str() {
                            "loading" | "interactive" => { saw_loading = true; }
                            "complete" if saw_loading || attempts > 3 => { break; }
                            _ => {}
                        }
                    }
                }
                Err(_) => { saw_loading = true; }
            }

            if attempts > 50 {
                info!(url, "page load timed out, proceeding with partial content");
                break;
            }
        }

        // Phase 2: Wait for SPA content to stabilize (React, Vue, etc.)
        // readyState=complete fires before JS frameworks render.
        // Strategy: wait until body length stabilizes AND no loading indicators are visible.
        let mut last_len: usize = 0;
        let mut stable_count = 0;
        for i in 0..6 {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            let len = match page.evaluate("document.body ? document.body.innerHTML.length : 0").await {
                Ok(r) => r.into_value::<usize>().unwrap_or(0),
                Err(_) => 0,
            };

            // Check for loading indicators — if found, reset stability counter
            let has_loader = match page.evaluate(DETECT_LOADING_JS).await {
                Ok(r) => r.into_value::<bool>().unwrap_or(false),
                Err(_) => false,
            };

            if has_loader {
                debug!(url, body_len = len, iteration = i, "loading indicator detected, waiting...");
                stable_count = 0;
                last_len = len;
                continue;
            }

            if len > 0 && (len.abs_diff(last_len) < 50) {
                stable_count += 1;
                if stable_count >= 2 {
                    debug!(url, body_len = len, "SPA content stabilized");
                    break;
                }
            } else {
                stable_count = 0;
            }
            last_len = len;
        }

        debug!(url, "page loaded, extracting DOM content");

        let _ = page.evaluate(SYNC_INPUT_VALUES_JS).await;

        let html = page
            .content()
            .await
            .map_err(|e| format!("Failed to get page content: {e}"))?;

        info!(url, html_len = html.len(), "got rendered HTML from browser");
        Ok(html)
    }

    pub async fn screenshot_page(&self) -> Result<Vec<u8>, String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let png = page
            .screenshot(ScreenshotParams::builder().full_page(true).build())
            .await
            .map_err(|e| format!("Screenshot failed: {e}"))?;

        info!(bytes = png.len(), "page screenshot captured");
        Ok(png)
    }

    pub async fn screenshot_element(&self, selector: &str) -> Result<Vec<u8>, String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let element = page
            .find_element(selector)
            .await
            .map_err(|e| format!("Element '{selector}' not found: {e}"))?;

        let png = element
            .screenshot(CaptureScreenshotFormat::Png)
            .await
            .map_err(|e| format!("Element screenshot failed: {e}"))?;

        info!(selector, bytes = png.len(), "element screenshot captured");
        Ok(png)
    }

    /// Check if the current page contains a CAPTCHA challenge.
    /// Returns a description of the CAPTCHA type if detected, or None.
    pub async fn detect_captcha(&self) -> Result<Option<String>, String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let js = r#"(() => {
            const signals = [];

            // reCAPTCHA v2/v3
            if (document.querySelector('.g-recaptcha, #recaptcha, [data-sitekey], iframe[src*="recaptcha"]')) {
                signals.push('reCAPTCHA');
            }

            // hCaptcha
            if (document.querySelector('.h-captcha, iframe[src*="hcaptcha"]')) {
                signals.push('hCaptcha');
            }

            // Cloudflare Turnstile
            if (document.querySelector('.cf-turnstile, iframe[src*="challenges.cloudflare"]')) {
                signals.push('Cloudflare Turnstile');
            }

            // Cloudflare challenge page (full-page interstitial)
            if (document.querySelector('#challenge-running, #challenge-form, .challenge-platform') ||
                document.title.includes('Just a moment') ||
                document.title.includes('Attention Required')) {
                signals.push('Cloudflare challenge page');
            }

            // Generic CAPTCHA detection
            if (document.querySelector('[class*="captcha" i], [id*="captcha" i], img[alt*="captcha" i]')) {
                if (signals.length === 0) signals.push('unknown CAPTCHA');
            }

            // AWS WAF / PerimeterX / DataDome
            if (document.querySelector('#px-captcha, .perimeterx, [src*="datadome"]')) {
                signals.push('bot protection CAPTCHA');
            }

            return signals.length > 0 ? signals.join(', ') : '';
        })()"#;

        match page.evaluate(js).await {
            Ok(result) => {
                if let Ok(val) = result.into_value::<String>() {
                    if val.is_empty() {
                        Ok(None)
                    } else {
                        info!(captcha = %val, "CAPTCHA detected on page");
                        Ok(Some(val))
                    }
                } else {
                    Ok(None)
                }
            }
            Err(e) => {
                debug!("CAPTCHA detection JS failed (page may be loading): {e}");
                Ok(None)
            }
        }
    }

    /// Wait for page to finish loading after a click that may have triggered navigation.
    /// Reuses the same wait logic as navigate(): readyState + body stabilization + loading indicators.
    pub async fn wait_for_page_stable(&self) -> Result<(), String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        // Phase 1: Wait for document.readyState === "complete"
        let mut attempts = 0;
        let mut saw_loading = false;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            attempts += 1;

            match page.evaluate("document.readyState").await {
                Ok(result) => {
                    if let Ok(state) = result.into_value::<String>() {
                        match state.as_str() {
                            "loading" | "interactive" => { saw_loading = true; }
                            "complete" if saw_loading || attempts > 3 => { break; }
                            _ => {}
                        }
                    }
                }
                Err(_) => { saw_loading = true; }
            }

            if attempts > 50 {
                info!("wait_for_page_stable: timed out");
                break;
            }
        }

        // Phase 2: Wait for SPA content to stabilize
        let mut last_len: usize = 0;
        let mut stable_count = 0;
        for i in 0..6 {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            let len = match page.evaluate("document.body ? document.body.innerHTML.length : 0").await {
                Ok(r) => r.into_value::<usize>().unwrap_or(0),
                Err(_) => 0,
            };

            let has_loader = match page.evaluate(DETECT_LOADING_JS).await {
                Ok(r) => r.into_value::<bool>().unwrap_or(false),
                Err(_) => false,
            };

            if has_loader {
                debug!(body_len = len, iteration = i, "loading indicator detected, waiting...");
                stable_count = 0;
                last_len = len;
                continue;
            }

            if len > 0 && (len.abs_diff(last_len) < 50) {
                stable_count += 1;
                if stable_count >= 2 {
                    debug!(body_len = len, "SPA content stabilized after click");
                    break;
                }
            } else {
                stable_count = 0;
            }
            last_len = len;
        }

        let _ = page.evaluate(SYNC_INPUT_VALUES_JS).await;
        Ok(())
    }

    pub async fn get_current_page_content(&self) -> Result<String, String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        // Sync JS input values into DOM attributes so they appear in .content()
        let _ = page.evaluate(SYNC_INPUT_VALUES_JS).await;

        let html = page
            .content()
            .await
            .map_err(|e| format!("Failed to get page content: {e}"))?;

        debug!(html_len = html.len(), "re-read page DOM content (with synced input values)");
        Ok(html)
    }

    /// Open the current page URL in the user's default browser via xdg-open.
    pub async fn open_in_browser(&self) -> Result<String, String> {
        let url = self.current_url().await?;
        info!(url = %url, "opening in default browser");
        std::process::Command::new("xdg-open")
            .arg(&url)
            .spawn()
            .map_err(|e| format!("Failed to open browser: {e}"))?;
        Ok(format!("Opened {url} in your default browser"))
    }

    pub async fn current_url(&self) -> Result<String, String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let url = page
            .url()
            .await
            .map_err(|e| format!("Failed to get URL: {e}"))?
            .ok_or("Page has no URL")?;

        Ok(url.to_string())
    }

    /// Count currently open pages/tabs.
    pub async fn page_count(&self) -> usize {
        self.browser.pages().await.map(|p| p.len()).unwrap_or(0)
    }

    /// If new tabs were opened (e.g. target="_blank" click), switch to the newest one
    /// and close the old tab. Returns true if we switched.
    pub async fn switch_to_newest_tab(&mut self, previous_count: usize) -> bool {
        // Brief pause for new tab to register
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let pages = match self.browser.pages().await {
            Ok(p) => p,
            Err(_) => return false,
        };

        if pages.len() <= previous_count {
            return false;
        }

        // The newest page is the last one
        if let Some(new_page) = pages.into_iter().last() {
            info!("click opened new tab — switching to it");
            self.current_page = Some(Arc::new(new_page));
            true
        } else {
            false
        }
    }

    pub async fn click(&self, selector: &str) -> Result<(), String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let sel_json = serde_json::to_string(selector)
            .unwrap_or_else(|_| format!("\"{}\"", selector));

        // Scroll element into view and check it exists
        let prep_js = format!(
            r#"(() => {{
                const el = document.querySelector({sel});
                if (!el) return "not_found";
                el.scrollIntoView({{block: "center", behavior: "instant"}});
                return "ok";
            }})()"#,
            sel = sel_json
        );

        if let Ok(result) = page.evaluate(prep_js).await {
            if let Ok(val) = result.into_value::<String>() {
                if val == "not_found" {
                    return Err(format!("Element '{selector}' not found"));
                }
            }
        }

        // Small delay after scrollIntoView for layout to settle
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Strategy 1: CDP-level click via chromiumoxide (dispatches real Input.dispatchMouseEvent)
        match page.find_element(selector).await {
            Ok(element) => {
                match element.click().await {
                    Ok(_) => {
                        debug!(selector, "clicked element via CDP");
                        return Ok(());
                    }
                    Err(e) => {
                        info!(selector, error = %e, "CDP click failed, trying JS fallback");
                    }
                }
            }
            Err(e) => {
                info!(selector, error = %e, "CDP find_element failed, trying JS fallback");
            }
        }

        // Strategy 2: JS click with full mouse event sequence
        let js = format!(
            r#"(() => {{
                let el = document.querySelector({sel});
                if (!el) return "not_found";
                el.scrollIntoView({{block: "center", behavior: "instant"}});
                const rect = el.getBoundingClientRect();
                const x = rect.left + rect.width / 2;
                const y = rect.top + rect.height / 2;
                const opts = {{bubbles: true, cancelable: true, view: window, clientX: x, clientY: y}};
                el.dispatchEvent(new MouseEvent("mouseover", opts));
                el.dispatchEvent(new MouseEvent("mousedown", opts));
                el.dispatchEvent(new MouseEvent("mouseup", opts));
                el.dispatchEvent(new MouseEvent("click", opts));
                if (el.tagName === "A" && el.href) {{
                    el.click();
                }}
                return "ok";
            }})()"#,
            sel = sel_json
        );

        match page.evaluate(js).await {
            Ok(result) => {
                if let Ok(val) = result.into_value::<String>() {
                    if val == "ok" {
                        debug!(selector, "clicked element via JS mouse events");
                        return Ok(());
                    }
                }
            }
            Err(_) => {}
        }

        Err(format!("Element '{selector}' not found"))
    }

    /// Click using a node's semantic description as fallback.
    /// Tries aria-label, button text content, and role-based matching via JS.
    pub async fn click_by_description(&self, description: &str, role_hint: Option<&str>) -> Result<(), String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        let desc_json = serde_json::to_string(description).unwrap_or_else(|_| "\"\"".into());
        let role_json = role_hint.map(|r| serde_json::to_string(r).unwrap_or_else(|_| "\"\"".into()))
            .unwrap_or_else(|| "null".into());

        let js = format!(
            r#"(() => {{
                const desc = {desc_json}.toLowerCase();
                const role = {role_json};

                // Strategy 1: aria-label match
                const allElements = document.querySelectorAll('[aria-label]');
                for (const el of allElements) {{
                    if (el.getAttribute('aria-label').toLowerCase().includes(desc)) {{
                        el.click();
                        return "aria:" + el.tagName;
                    }}
                }}

                // Strategy 2: button/link text content match
                const clickables = document.querySelectorAll('button, a, [role="button"]');
                for (const el of clickables) {{
                    const text = (el.textContent || '').trim().toLowerCase();
                    if (text && desc.includes(text)) {{
                        el.click();
                        return "text:" + el.tagName;
                    }}
                }}

                // Strategy 3: role-based (e.g. dialog close button)
                if (role) {{
                    const roleEls = document.querySelectorAll('[role="' + role + '"] button, [role="' + role + '"] [role="button"]');
                    for (const el of roleEls) {{
                        const label = el.getAttribute('aria-label') || el.textContent || '';
                        if (label.toLowerCase().includes('close') || label.toLowerCase().includes('dismiss')) {{
                            el.click();
                            return "role:" + el.tagName;
                        }}
                    }}
                }}

                return "not_found";
            }})()"#
        );

        match page.evaluate(js).await {
            Ok(result) => {
                if let Ok(val) = result.into_value::<String>() {
                    if val != "not_found" {
                        info!(description, matched = %val, "clicked element by description");
                        return Ok(());
                    }
                }
            }
            Err(e) => {
                return Err(format!("JS click evaluation failed: {e}"));
            }
        }

        Err(format!("No clickable element matching description '{description}' found"))
    }

    pub async fn fill(&self, selector: &str, text: &str) -> Result<(), String> {
        let page = self.current_page.as_ref()
            .ok_or("No page loaded")?;

        page.find_element(selector)
            .await
            .map_err(|e| format!("Element '{selector}' not found: {e}"))?
            .click()
            .await
            .map_err(|e| format!("Click failed on '{selector}': {e}"))?
            .type_str(text)
            .await
            .map_err(|e| format!("Type failed on '{selector}': {e}"))?;

        debug!(selector, text_len = text.len(), "typed text into element");
        Ok(())
    }

    pub async fn close(mut self) -> Result<(), String> {
        info!("closing browser...");
        self.browser
            .close()
            .await
            .map_err(|e| format!("Failed to close browser: {e}"))?;
        Ok(())
    }
}
