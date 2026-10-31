"""Capture real functioning dashboard offline (no network needed)."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import base64,json,re
root=Path(__file__).resolve().parents[1]
w=root/'web';out=root/'docs/screenshots';out.mkdir(exist_ok=True,parents=True)
html=(w/'index.html').read_text()
css=(w/'styles.css').read_text();js=(w/'app.js').read_text()
res=json.loads((w/'assets/results.json').read_text())
samples=json.loads((w/'assets/showcase.json').read_text())
def datauri(p): return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
for s in samples:
 for k in ['original_image','cnn_image','resnet_image']:
  s[k]=datauri(w/s[k])
html=html.replace('src="assets/hero-attribution.png"','src="'+datauri(w/'assets/hero-attribution.png')+'"')
html=re.sub(r'<link[^>]*>', '',html)
html=html.replace('</head>','<style>'+css+'</style></head>')
# The offline renderer uses the same app.js with the real experiment JSON.
shim='window.fetch = async function(path){ return {ok: true,json: async()=>path.includes("results") ? window.__metrics: window.__samples};};\nwindow.__metrics='+json.dumps(res)+';\nwindow.__samples='+json.dumps(samples)+';'
html=html.replace('<script src="app.js" defer></script>','<script>'+shim+'</script><script>'+js+'</script>')
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox','--disable-dev-shm-usage','--disable-web-security'])
    page=browser.new_page(viewport={'width':1440,'height':950},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content(html,wait_until='load',timeout=30000)
    page.wait_for_function("document.getElementById('cnn-accuracy').textContent.includes('%')",timeout=10000)
    page.screenshot(path=str(out/'01_dashboard_hero.png'),animations='disabled')
    page.screenshot(path=str(out/'02_full_research_dashboard.png'),full_page=True,animations='disabled')
    page.locator('#comparison').screenshot(path=str(out/'03_model_comparison.png'))
    page.locator('#explainability').screenshot(path=str(out/'04_gradcam_workbench.png'))
    page.locator('.view-mode[data-mode="cnn"]').click()
    page.locator('#explainability').screenshot(path=str(out/'05_custom_cnn_explanations.png'))
    page.locator('.sample-card').nth(8).click()
    page.locator('#explainability').screenshot(path=str(out/'06_interactive_test_case.png'))
    assert len(page.locator('.sample-card').all())==12
    assert not errors,errors
    print('Screenshots',[(x.name,round(x.stat().st_size/1024)) for x in out.glob('*.png')]);print('Page errors',errors)
    browser.close()
